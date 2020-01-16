"""
Converts raw images taken through a spectroscope into R, G, and B spectral sensativity curves
Intended to be run as a script
"""
import cv2
import rawpy
import numpy as np
import spectrumTools

#[Image Name, Threshold (can be manually tweaked. helps with processing), [Start Wavelength, End Wavelength]]
skyImg = ['sky', 0.5, [385, 725]]
sunImg = ['sunlightDim', 0.08, [385, 725]]
benQImg = ['BenQ2', 0.1, [385, 725]]
iPadImg = ['iPad', 0.3, [385, 725]]
incadecentAImg = ['IncadecentA_card', 0.5, [385, 725]]
incadecentBImg = ['IncadecentB_card', 0.5, [385, 725]]
ledImg = ['LED_card', 0.5, [385, 725]]

def stretch(img, mask=None):
    """Stretch the image to enhance contrast"""
    mask = mask if mask is not None else np.ones(img.shape, dtype='bool')
    minVal = np.min(img[mask])
    maxVal = np.max(img[mask])

    magnitude = maxVal - minVal
    stretched = (img - minVal) / magnitude
    stretched[np.logical_not(mask)] = 0
    return stretched

def autoBB(img, threshold):
    """Return the BB for the wavelength numbers and for the spectrum"""
    scaled = stretch(img)

    brightSubpixelMask = scaled > threshold
    brightSubpixelMask = brightSubpixelMask.astype('uint8') * 255

    morphologyKernel = np.ones((21, 21), np.uint8)
    dilated = cv2.dilate(brightSubpixelMask, morphologyKernel, iterations=1)

    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(contour) for contour in contours]
    largestContour = np.argmax(areas)

    numbersBB = cv2.boundingRect(contours[largestContour])
    spectrumBB = numbersBB + np.array([0, int(1 * numbersBB[3]), 0, int(-0.5 * numbersBB[3])]) #Just offset the numberline by 2 times its height. Samples roughly the middle of the spectrum area

    return [numbersBB, spectrumBB]

def extractSpectrum(spectrum, mask):
    """
    Takes a image of just the spectroscope spectrum and a mask defining which color pixels to extract
        returns both the masked spectrum image and the median of each column of pixels. Each column should correlate with a certain wavlength range (to be calculated later)
    """
    img = np.copy(spectrum)
    img[np.logical_not(mask)] = 0

    rowMask = np.any(img.T, axis=0)
    medians = np.median(img.T[:, rowMask], axis=1)
    medians = medians[medians > 0]
    return [img, medians]

#Crop is [[Y, X], [Height, Width], HeightRatio]
def extractSpectrums(imgFileName, threshold, wavelengthRange):#, crop=None):
    """Extracts the spectrum and returns an object containing the wavelength number image, each color channel image, each color channel medians, and the wavelength range"""
    with rawpy.imread('images/imagesRed/{}.DNG'.format(imgFileName)) as raw:
        imgDim = raw.raw_image.shape

        img = raw.raw_image
        colors = raw.raw_colors

    redMask = colors == 0
    greenMask = colors == 1
    blueMask = colors == 2

    numbersBB, spectrumBB = autoBB(img, threshold)

    numbers = np.copy(img[numbersBB[1]:(numbersBB[1] + numbersBB[3]), numbersBB[0]:(numbersBB[0] + numbersBB[2])])
    spectrum = np.copy(img[spectrumBB[1]:(spectrumBB[1] + spectrumBB[3]), spectrumBB[0]:(spectrumBB[0] + spectrumBB[2])])

    numbersRedMask = redMask[numbersBB[1]:(numbersBB[1] + numbersBB[3]), numbersBB[0]:(numbersBB[0] + numbersBB[2])]

    redMask = redMask[spectrumBB[1]:(spectrumBB[1] + spectrumBB[3]), spectrumBB[0]:(spectrumBB[0] + spectrumBB[2])]
    greenMask = greenMask[spectrumBB[1]:(spectrumBB[1] + spectrumBB[3]), spectrumBB[0]:(spectrumBB[0] + spectrumBB[2])]
    blueMask = blueMask[spectrumBB[1]:(spectrumBB[1] + spectrumBB[3]), spectrumBB[0]:(spectrumBB[0] + spectrumBB[2])]

    print('Numbers Size :: {}'.format(numbersBB))
    print('Specturm Size :: {}'.format(spectrumBB))

    numbers = stretch(numbers, numbersRedMask)
    stretched = stretch(spectrum)

    redImg, redMedians = extractSpectrum(stretched, redMask)
    greenImg, greenMedians = extractSpectrum(stretched, greenMask)
    blueImg, blueMedians = extractSpectrum(stretched, blueMask)

    lenRed = len(redMedians)
    lenGreen = len(greenMedians)
    lenBlue = len(blueMedians)

    offByOneCheck = min([lenRed, lenGreen, lenBlue])
    if ((lenRed - offByOneCheck) > 1) or ((lenGreen - offByOneCheck) > 1) or ((lenBlue - offByOneCheck) > 1):
        raise ValueError('Number of pixels offset > 1')

    redMedians = redMedians[:offByOneCheck]
    greenMedians = greenMedians[:offByOneCheck]
    blueMedians = blueMedians[:offByOneCheck]

    return [numbers, [redImg, redMedians], [greenImg, greenMedians], [blueImg, blueMedians], wavelengthRange]

def showSpectrum(imageSpectrumObject, name, wait=True):
    """Show the spectrum that extractSpectrums returns. Breaks the spectrums out by RGB"""
    numbers, red, green, blue, _ = imageSpectrumObject

    stacked = np.vstack([numbers, red[0], green[0], blue[0]])
    cv2.imshow('RGB {}'.format(name), stacked)
    if wait:
        cv2.waitKey(0)

def getCurves(imageSpectrumObject):
    """Takes the output of extract spectrums and returns the RGB curve object"""
    _, red, green, blue, wavelengthRange = imageSpectrumObject

    redX = np.linspace(wavelengthRange[0], wavelengthRange[1], len(red[1]))
    greenX = np.linspace(wavelengthRange[0], wavelengthRange[1], len(green[1]))
    blueX = np.linspace(wavelengthRange[0], wavelengthRange[1], len(blue[1]))

    redCurve = np.stack([redX, red[1]], axis=1)
    greenCurve = np.stack([greenX, green[1]], axis=1)
    blueCurve = np.stack([blueX, blue[1]], axis=1)

    redCurveObject = spectrumTools.makeCurveObject(redCurve, wavelengthRange, [0, 1], [0, 1])
    greenCurveObject = spectrumTools.makeCurveObject(greenCurve, wavelengthRange, [0, 1], [0, 1])
    blueCurveObject = spectrumTools.makeCurveObject(blueCurve, wavelengthRange, [0, 1], [0, 1])

    return [redCurveObject, greenCurveObject, blueCurveObject]


def saveCurves(rgbSpectrumList, name):
    """Saves the RGB curve object"""
    spectrumTools.writeMeasuredCurve('{}_red'.format(name), rgbSpectrumList[0])
    spectrumTools.writeMeasuredCurve('{}_green'.format(name), rgbSpectrumList[1])
    spectrumTools.writeMeasuredCurve('{}_blue'.format(name), rgbSpectrumList[2])


led = extractSpectrums(*ledImg)
ledCurves = getCurves(led)
spectrumTools.plotRGBCurves(ledCurves)
#saveCurves(ledCurves, 'led')
showSpectrum(led, 'led')

#incA = extractSpectrums(*incadecentAImg)
#incACurves = getCurves(incA)
#spectrumTools.plotRGBCurves(incACurves)
#saveCurves(incACurves, 'incA')
#showSpectrum(incA, 'incA')

#incB = extractSpectrums(*incadecentBImg)
#incBCurves = getCurves(incB)
#spectrumTools.plotRGBCurves(incBCurves)
#saveCurves(incBCurves, 'incB')
#showSpectrum(incB, 'incB')

#sky = extractSpectrums(*skyImg)
#skyCurves = getCurves(sky)
#spectrumTools.plotRGBCurves(skyCurves)
#saveCurves(skyCurves, 'sky')
#showSpectrum(sky, 'sky')

#sun = extractSpectrums(*sunImg)
#sunCurves = getCurves(sun)
#spectrumTools.plotRGBCurves(sunCurves)
#saveCurves(sunCurves, 'sun')
#showSpectrum(sun, 'sun')

#benQ = extractSpectrums(*benQImg)
#benQCurves = getCurves(benQ)
#spectrumTools.plotRGBCurves(benQCurves)
#saveCurves(benQCurves, 'benQ')
#showSpectrum(benQ, 'benQ')

#iPad = extractSpectrums(*iPadImg)
#iPadCurves = getCurves(iPad)
#spectrumTools.plotRGBCurves(iPadCurves)
#saveCurves(iPadCurves, 'iPad')
#showSpectrum(iPad, 'iPad')
