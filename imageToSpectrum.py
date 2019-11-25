import cv2
import rawpy
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import math

skyImg = ['sky', 0.5]
sunDimImg = ['sunlightDim', 0.08]
benQImg = ['BenQ2', 0.1]
iPadImg = ['iPad', 0.3] #400nm -> 710nm
incadecentAImg = ['IncadecentA_card', 0.5]
incadecentBImg = ['IncadecentB_card', 0.5]
ledImg = ['LED_card' , 0.5]

def stretch(img, mask=None):
    mask = mask if mask is not None else np.ones(img.shape, dtype='bool')
    minVal = np.min(img[mask])
    maxVal = np.max(img[mask])

    magnitude = maxVal - minVal
    stretched = (img - minVal) / magnitude
    stretched[np.logical_not(mask)] = 0
    return stretched#, magnitude

def autoBB(img, threshold, redMask, greenMask, blueMask):
    scaled = stretch(img)

    brightSubpixelMask = scaled > threshold
    brightSubpixelMask = brightSubpixelMask.astype('uint8') * 255

    morphologyKernel = np.ones((21, 21), np.uint8)
    dilated = cv2.dilate(brightSubpixelMask, morphologyKernel, iterations=1)

    contours, heirarchy = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(contour) for contour in contours]
    largestContour = np.argmax(areas)

    numbersBB = cv2.boundingRect(contours[largestContour])
    #spectrumBB = numbersBB + np.array([0, int(1.5 * numbersBB[3]), 0, 0)]) #Just offset the numberline by 2 times its height. Samples roughly the middle of the spectrum area
    spectrumBB = numbersBB + np.array([0, int(1 * numbersBB[3]), 0, int(-0.5 * numbersBB[3])]) #Just offset the numberline by 2 times its height. Samples roughly the middle of the spectrum area

    return [numbersBB, spectrumBB]

def extractSpectrum(spectrum, mask, stretch=True):
    img = np.copy(spectrum)
    if stretch:
        img = stretch(img, mask)
    else:
        img[np.logical_not(mask)] = 0
    rowMask = np.any(img.T, axis=0)
    medians = np.median(img.T[:, rowMask], axis=1)
    medians = medians[medians > 0]
    return [img, medians]


def amplify(img):
    std = np.std(img)
    median = np.median(img)
    floor = median + (0.5 * std)
    img[img > floor] = 1
    return img

def loadRawImage(filename):
    with rawpy.imread('imagesRed/{}.DNG'.format(imgFileName)) as raw:
        imgDim = raw.raw_image.shape

        img = raw.raw_image
        colors = raw.raw_colors


#Crop is [[Y, X], [Height, Width], HeightRatio]
def extractSpectrums(imgFileName, threshold):#, crop=None):
    with rawpy.imread('images/imagesRed/{}.DNG'.format(imgFileName)) as raw:
        imgDim = raw.raw_image.shape

        img = raw.raw_image
        colors = raw.raw_colors

    
    redMask = colors == 0
    #greenMask = np.logical_or((raw.raw_colors == 1), (raw.raw_colors == 3))
    greenMask = colors == 1
    blueMask = colors == 2
        
    numbersBB, spectrumBB = autoBB(img, threshold, redMask, greenMask, blueMask)

    numbers = np.copy(img[numbersBB[1]:(numbersBB[1] + numbersBB[3]), numbersBB[0]:(numbersBB[0] + numbersBB[2])])
    spectrum = np.copy(img[spectrumBB[1]:(spectrumBB[1] + spectrumBB[3]), spectrumBB[0]:(spectrumBB[0] + spectrumBB[2])])

    numbersRedMask = redMask[numbersBB[1]:(numbersBB[1] + numbersBB[3]), numbersBB[0]:(numbersBB[0] + numbersBB[2])]

    redMask = redMask[spectrumBB[1]:(spectrumBB[1] + spectrumBB[3]), spectrumBB[0]:(spectrumBB[0] + spectrumBB[2])]
    greenMask = greenMask[spectrumBB[1]:(spectrumBB[1] + spectrumBB[3]), spectrumBB[0]:(spectrumBB[0] + spectrumBB[2])]
    blueMask = blueMask[spectrumBB[1]:(spectrumBB[1] + spectrumBB[3]), spectrumBB[0]:(spectrumBB[0] + spectrumBB[2])]

    #if crop is None:
    #    numbersBB, spectrumBB = autocrop(img, threshold, redMask, greenMask, blueMask)

    #    numbers = np.copy(img[numbersBB[1]:(numbersBB[1] + numbersBB[3]), numbersBB[0]:(numbersBB[0] + numbersBB[2])])
    #    spectrum = np.copy(img[spectrumBB[1]:(spectrumBB[1] + spectrumBB[3]), spectrumBB[0]:(spectrumBB[0] + spectrumBB[2])])

    #    numbersRedMask = redMask[numbersBB[1]:(numbersBB[1] + numbersBB[3]), numbersBB[0]:(numbersBB[0] + numbersBB[2])]

    #    redMask = redMask[spectrumBB[1]:(spectrumBB[1] + spectrumBB[3]), spectrumBB[0]:(spectrumBB[0] + spectrumBB[2])]
    #    greenMask = greenMask[spectrumBB[1]:(spectrumBB[1] + spectrumBB[3]), spectrumBB[0]:(spectrumBB[0] + spectrumBB[2])]
    #    blueMask = blueMask[spectrumBB[1]:(spectrumBB[1] + spectrumBB[3]), spectrumBB[0]:(spectrumBB[0] + spectrumBB[2])]

    #else:

    #    print('crop :: {}'.format(crop))
    #    print('dim :: {}'.format(imgDim))
    #    height, width = imgDim

    #    numbersBB = np.round(np.array(crop[0:2]) * np.array([height, width])).astype('int32')
    #    print('Numbers BB :: {}'.format(numbersBB))

    #    spectrumY = np.round(numbersBB[0, 0] + numbersBB[1, 0])
    #    print('Spectrum Y :: {}'.format(spectrumY))

    #    spectrumHeight = np.round(numbersBB[1, 0] * crop[2])
    #    print('Spectrum Height :: {}'.format(spectrumHeight))

    #    spectrumBB = np.array([[spectrumY, numbersBB[0, 1]], [spectrumHeight, numbersBB[1][1]]]).astype('int32')
    #    print('Spectrum BB :: {}'.format(spectrumBB))

    #    numbersBB = numbersBB.flatten()
    #    spectrumBB = spectrumBB.flatten()

    #    numbers = np.copy(img[numbersBB[0]:(numbersBB[0] + numbersBB[2]), numbersBB[1]:(numbersBB[1] + numbersBB[3])])
    #    spectrum = np.copy(img[spectrumBB[0]:(spectrumBB[0] + spectrumBB[2]), spectrumBB[1]:(spectrumBB[1] + spectrumBB[3])])

    #    numbersRedMask = redMask[numbersBB[0]:(numbersBB[0] + numbersBB[2]), numbersBB[1]:(numbersBB[1] + numbersBB[3])]

    #    redMask = redMask[spectrumBB[0]:(spectrumBB[0] + spectrumBB[2]), spectrumBB[1]:(spectrumBB[1] + spectrumBB[3])]
    #    greenMask = greenMask[spectrumBB[0]:(spectrumBB[0] + spectrumBB[2]), spectrumBB[1]:(spectrumBB[1] + spectrumBB[3])]
    #    blueMask = blueMask[spectrumBB[0]:(spectrumBB[0] + spectrumBB[2]), spectrumBB[1]:(spectrumBB[1] + spectrumBB[3])]



    #numbersBB, spectrumBB = autocrop(img, threshold, redMask, greenMask, blueMask)
    print('Numbers Size :: {}'.format(numbersBB))
    print('Specturm Size :: {}'.format(spectrumBB))


    #numbers = stretch(numbers)
    #spectrum = stretch(spectrum)
    #cv2.imshow('numbers', numbers)
    #cv2.imshow('spectrum', spectrum)
    #cv2.waitKey(0)

    #numbers = stretch(numbers)
    #divide = np.ones([1, numbers.shape[1]])
    #stacked = np.vstack([numbers, divide, spectrum])

    #cv2.imshow('check', stretch(stacked))
    #cv2.waitKey(0)


    numbers = stretch(numbers, numbersRedMask)


    stretched = stretch(spectrum)

    #redImg, redMagnitude, redMedians = extractSpectrum(spectrum, redMask, False)
    redImg, redMedians = extractSpectrum(stretched, redMask, False)
    #print('Red Magnitude :: {}'.format(redMagnitude))

    #greenImg, greenMagnitude, greenMedians = extractSpectrum(spectrum, greenMask, False)
    greenImg, greenMedians = extractSpectrum(stretched, greenMask, False)
    #print('Green Magnitude :: {}'.format(greenMagnitude))

    #blueImg, blueMagnitude, blueMedians = extractSpectrum(spectrum, blueMask, False)
    blueImg, blueMedians = extractSpectrum(stretched, blueMask, False)
    #print('Blue Magnitude :: {}'.format(blueMagnitude))

    lenRed = len(redMedians)
    lenGreen = len(greenMedians)
    lenBlue = len(blueMedians)

    offByOneCheck = min([lenRed, lenGreen, lenBlue])
    if ((lenRed - offByOneCheck) > 1) or ((lenGreen - offByOneCheck) > 1) or ((lenBlue - offByOneCheck) > 1):
        raise ValueError('Number of pixels offset > 1')

    redMedians = redMedians[:offByOneCheck]
    greenMedians = greenMedians[:offByOneCheck]
    blueMedians = blueMedians[:offByOneCheck]

    return [numbers, [redImg, redMedians], [greenImg, greenMedians], [blueImg, blueMedians]]

def showSpectrum(imageSpectrumObject, name, wait=True):
    numbers, red, green, blue = imageSpectrumObject
    
    stacked = np.vstack([numbers, red[0], green[0], blue[0]])
    cv2.imshow('RGB {}'.format(name), stacked)
    if wait:
        cv2.waitKey(0)

led = extractSpectrums(*ledImg)
showSpectrum(led, 'led')

#incA = extractSpectrums(incadecentAImg, 0.5)
#showSpectrum(incA, 'Inc A', False)

#incB = extractSpectrums(incadecentBImg, 0.5)
#showSpectrum(incB, 'Inc B', False)

#sky = extractSpectrums(skyImg, 0.5)
#showSpectrum(sky, 'sky', False)

#sun = extractSpectrums(sunDimImg, 0.08)
#showSpectrum(sun, 'Sun', False)

#benQ = extractSpectrums(benQImg, 0.1)
#showSpectrum(benQ, 'ben Q', False)

#iPad = extractSpectrums(iPadImg, 0.3)
#showSpectrum(iPad, 'iPad')
