import cv2
import rawpy
import numpy as np
import matplotlib.pyplot as plt


#skyImg = 'sky5'
##imgName = 'sunRef2' #Possibly Perfect
#sunImg = 'sunRef3' #Possibly Perfect
###imgName = 'sunRef4' #Under Exposed
###imgName = 'sunRef5' #Under Exposed
###imgName = 'sunRef6' #Over Exposed
##imgName = 'sunRef7' #Over Exposed
#iPadImg = 'iPad1'
##imgName = 'iPad_ScreenFlash1'
##imgName = 'benQ1'
#stoveImg = 'StoveLight'
#skyImg = 'sky'
#skyImg = 'purpleDim'
#skyImg = 'purpleBright'
sunDimImg = 'sunlightDim' 
sunMediumImg = 'sunlightMedium'#395nm -> 700nm
#sunDimImg = 'sunlightDim'
benQImg = 'BenQ2'
iPadImg = 'iPad' #400nm -> 710nm

blueEyeCurve = np.array([[0,3],[9,10],[12,18],[24,91],[25,115],[39,181],[42,189],[43,189],[45,180],[52,157],[55,152],[58,131],[61,125],[77,41],[87,20],[92,16],[95,10],[109,2],[114,1],[120,0],[256,0]])
greenEyeCurve = np.array([[0,0],[15,3],[22,5],[27,7],[48,16],[55,21],[66,41],[76,56],[84,74],[91,108],[98,147],[108,181],[114,188],[116,189],[125,188],[127,186],[128,182],[138,158],[140,157],[157,96],[159,80],[168,50],[173,35],[185,14],[192,7],[210,1],[211,0],[256,0]])
redEyeCurve = np.array([[0,0],[25,3],[45,8],[58,14],[74,30],[82,45],[87,60],[95,92],[103,130],[109,150],[114,159],[124,177],[128,183],[131,183],[134,184],[139,188],[143,189],[145,188],[148,185],[149,183],[152,182],[157,175],[168,148],[179,110],[192,63],[201,35],[205,25],[213,15],[222,7],[230,3],[240,0], [256,0]])


def stretch(img, mask=None):
    mask = mask if mask is not None else np.ones(img.shape, dtype='bool')
    minVal = np.min(img[mask])
    maxVal = np.max(img[mask])

    magnitude = maxVal - minVal
    stretched = (img - minVal) / magnitude
    stretched[np.logical_not(mask)] = 0
    return stretched#, magnitude

def autocrop(img, threshold, redMask, greenMask, blueMask):
    scaled = stretch(img)

    #med = np.median(scaled[scaled > (10/255)])
    #print('Med :: {}'.format(med))

    #threshold = 0.30
    #threshold = 0.50
    brightSubpixelMask = scaled > threshold
    brightSubpixelMask = brightSubpixelMask.astype('uint8') * 255

    morphologyKernel = np.ones((21, 21), np.uint8)
    dilated = cv2.dilate(brightSubpixelMask, morphologyKernel, iterations=1)

    contours, heirarchy = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(contour) for contour in contours]
    largestContour = np.argmax(areas)

    numbersBB = cv2.boundingRect(contours[largestContour])
    #spectrumBB = numbersBB + np.array([0, int(1.5 * numbersBB[3]), 0, 0)]) #Just offset the numberline by 2 times its height. Samples roughly the middle of the spectrum area
    spectrumBB = numbersBB + np.array([0, int(1.75 * numbersBB[3]), 0, int(-0.5 * numbersBB[3])]) #Just offset the numberline by 2 times its height. Samples roughly the middle of the spectrum area

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

def extractSpectrums(imgFileName, threshold):
    with rawpy.imread('imagesRed/{}.DNG'.format(imgFileName)) as raw:
        redMask = raw.raw_colors == 0
        #greenMask = np.logical_or((raw.raw_colors == 1), (raw.raw_colors == 3))
        greenMask = raw.raw_colors == 1
        blueMask = raw.raw_colors == 2

        numbersBB, spectrumBB = autocrop(raw.raw_image, threshold, redMask, greenMask, blueMask)

        numbers = np.copy(raw.raw_image[numbersBB[1]:(numbersBB[1] + numbersBB[3]), numbersBB[0]:(numbersBB[0] + numbersBB[2])])
        spectrum = np.copy(raw.raw_image[spectrumBB[1]:(spectrumBB[1] + spectrumBB[3]), spectrumBB[0]:(spectrumBB[0] + spectrumBB[2])])

    numbersRedMask = redMask[numbersBB[1]:(numbersBB[1] + numbersBB[3]), numbersBB[0]:(numbersBB[0] + numbersBB[2])]

    redMask = redMask[spectrumBB[1]:(spectrumBB[1] + spectrumBB[3]), spectrumBB[0]:(spectrumBB[0] + spectrumBB[2])]
    greenMask = greenMask[spectrumBB[1]:(spectrumBB[1] + spectrumBB[3]), spectrumBB[0]:(spectrumBB[0] + spectrumBB[2])]
    blueMask = blueMask[spectrumBB[1]:(spectrumBB[1] + spectrumBB[3]), spectrumBB[0]:(spectrumBB[0] + spectrumBB[2])]

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

    return [numbers, [redImg, redMedians], [greenImg, greenMedians], [blueImg, blueMedians]]

def spectrumCheck(imgFileName, threshold):
    numbers, red, green, blue = extractSpectrums(imgFileName, threshold)

    fig = plt.figure()
    ax1 = fig.add_axes([0.1, 0.5, 0.85, 0.5])
    ax1.plot(np.arange(len(red[1])), red[1], 'r-')
    ax1.plot(np.arange(len(green[1])), green[1], 'g-')
    ax1.plot(np.arange(len(blue[1])), blue[1], 'b-')

    ax2 = fig.add_axes([0.1, 0, 0.85, 0.5])
    ax2.imshow(np.vstack([numbers, red[0], green[0], blue[0]]), cmap='gray')
    #ax2.imshow(numbers, cmap='gray')

    fig.show()

def scalePoints(red, green, blue, startWavelength, endWavelength, scaleIndependently):
    wavelengthRange = endWavelength - startWavelength

    redMaxes = np.max(red, axis=0)
    greenMaxes = np.max(green, axis=0)
    blueMaxes = np.max(blue, axis=0)

    if not scaleIndependently:
        totalMax = np.max([redMaxes, greenMaxes, blueMaxes], axis=0)
        redMaxes = totalMax
        greenMaxes = totalMax
        blueMaxes = totalMax

    scaledRed = red / redMaxes
    scaledRed[:, 0] = startWavelength + (scaledRed[:, 0] * wavelengthRange)
    quantizedRed = quantizeSpectrum(scaledRed)

    scaledGreen = green / greenMaxes
    scaledGreen[:, 0] = startWavelength + (scaledGreen[:, 0] * wavelengthRange)
    quantizedGreen = quantizeSpectrum(scaledGreen)

    scaledBlue = blue / blueMaxes
    scaledBlue[:, 0] = startWavelength + (scaledBlue[:, 0] * wavelengthRange)
    quantizedBlue = quantizeSpectrum(scaledBlue)

    return [quantizedRed, quantizedGreen, quantizedBlue]

#y=mx+b
#m = rise/run... (y2 - y1) / (x2 - x1)
#b = y1 - m * x1
def quantizeSpectrum(points):

    sortedPoints = np.array(sorted(points, key=lambda point: point[0]))
    pointDiffs = sortedPoints[1:] - sortedPoints[:-1]

    slopes = pointDiffs[:, 1] / pointDiffs[:, 0]
    intercepts = sortedPoints[:-1, 1] - (slopes * sortedPoints[:-1, 0])
    slopeRanges = np.stack([sortedPoints[:-1, 0], sortedPoints[1:, 0]], axis=1)

    points = []
    for m, b, lineRange in zip(slopes, intercepts, slopeRanges):
        if (lineRange[0] == lineRange[1]) or (lineRange[0] == (lineRange[1] + 1)):
            xValues = lineRange[0]
        else:
            xValues = np.arange(np.floor(lineRange[0]), np.floor(lineRange[1]))
        yValues = xValues * m + b
        points.append(np.stack([xValues, yValues], axis=1))

    points = np.concatenate(points, axis=0)
    return points

def combineRGBtoFullSpectrum(red, green, blue):
    combined = red


def plotWithScale(red, green, blue, startWavelength, endWavelength):

    scaledRed, scaledGreen, scaledBlue = scalePoints(red, green, blue, startWavelength, endWavelength)

    plt.plot(scaledBlue[:, 0], scaledBlue[:, 1], 'b-')
    plt.plot(scaledGreen[:, 0], scaledGreen[:, 1], 'g-')
    plt.plot(scaledRed[:, 0], scaledRed[:, 1], 'r-')
    plt.show()

iPad = extractSpectrums(iPadImg, 0.30)
iPadRed = iPad[1][1]
iPadRed = np.stack([np.arange(len(iPadRed)), np.array(iPadRed)], axis=1)

iPadGreen = iPad[2][1]
iPadGreen = np.stack([np.arange(len(iPadGreen)), np.array(iPadGreen)], axis=1)

iPadBlue = iPad[3][1]
iPadBlue = np.stack([np.arange(len(iPadBlue)), np.array(iPadBlue)], axis=1)

sunlight = extractSpectrums(sunMediumImg, 0.50)
sunlightRed = sunlight[1][1]
sunlightRed = np.stack([np.arange(len(sunlightRed)), np.array(sunlightRed)], axis=1)

sunlightGreen = sunlight[2][1]
sunlightGreen = np.stack([np.arange(len(sunlightGreen)), np.array(sunlightGreen)], axis=1)

sunlightBlue = sunlight[3][1]
sunlightBlue = np.stack([np.arange(len(sunlightBlue)), np.array(sunlightBlue)], axis=1)

#quantizeSpectrum(redEyeCurve)

#redEyePoints = quantizeSpectrum(redEyeCurve)
#greenEyePoints = quantizeSpectrum(greenEyeCurve)
#blueEyePoints = quantizeSpectrum(blueEyeCurve)

#plotWithScale(redEyeCurve, greenEyeCurve, blueEyeCurve, 390, 700)
#plotWithScale(redEyePoints, greenEyePoints, blueEyePoints, 390, 700)
#plotWithScale(iPadRed, iPadGreen, iPadBlue, 400, 710)
#plotWithScale(sunlightRed, sunlightGreen, sunlightBlue, 395, 700)

scaledEyeSensitivity = scalePoints(redEyeCurve, greenEyeCurve, blueEyeCurve, 390, 700, True)
scaledIpadLightEmission = scalePoints(iPadRed, iPadGreen, iPadBlue, 390, 700, True)
scaledSunlightEmission = scalePoints(sunlightRed, sunlightGreen, sunlightBlue, 395, 700, True)

#plt.plot(scaledSunlightEmission[0][:, 0], scaledSunlightEmission[0][:, 1], 'r-')
#plt.plot(scaledSunlightEmission[1][:, 0], scaledSunlightEmission[1][:, 1], 'g-')
#plt.plot(scaledSunlightEmission[2][:, 0], scaledSunlightEmission[2][:, 1], 'b-')

plt.plot(scaledIpadLightEmission[0][:, 0], scaledIpadLightEmission[0][:, 1], 'r--')
plt.plot(scaledIpadLightEmission[1][:, 0], scaledIpadLightEmission[1][:, 1], 'g--')
plt.plot(scaledIpadLightEmission[2][:, 0], scaledIpadLightEmission[2][:, 1], 'b--')

plt.plot(scaledEyeSensitivity[0][:, 0], scaledEyeSensitivity[0][:, 1], 'r-')
plt.plot(scaledEyeSensitivity[1][:, 0], scaledEyeSensitivity[1][:, 1], 'g-')
plt.plot(scaledEyeSensitivity[2][:, 0], scaledEyeSensitivity[2][:, 1], 'b-')
plt.show()

print('Scaled :: {}'.format(scaledSunlightEmission))


#spectrumCheck(sunDimImg, 0.30)
#spectrumCheck(sunMediumImg, 0.50)
#spectrumCheck(benQImg, 0.10)
#spectrumCheck(iPadImg, 0.30)
cv2.imshow('Force Pause', np.ones([100, 100, 1], dtype='uint8'))
cv2.waitKey(0)
