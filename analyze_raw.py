import cv2
import random
import rawpy
import numpy as np
import matplotlib.pyplot as plt
#from scipy.interpolate import splev, splrep
from scipy.signal import savgol_filter
import math

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
skyImg = 'sky'
#skyImg = 'purpleDim'
#skyImg = 'purpleBright'
sunDimImg = 'sunlightDim' 
#sunMediumImg = 'sunlightMedium'#395nm -> 700nm
#sunDimImg = 'sunlightDim'
benQImg = 'BenQ2'
iPadImg = 'iPad' #400nm -> 710nm

incadecentAImg = 'IncadecentA'
incadecentBImg = 'IncadecentB'
incadecentBBrightImg = 'IncadecentB_bright'
ledImg = 'LED' 

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


def amplify(img):
    std = np.std(img)
    median = np.median(img)
    floor = median + (0.5 * std)
    img[img > floor] = 1
    return img

#Crop is [[Y, X], [Height, Width], HeightRatio]
def extractSpectrums(imgFileName, threshold, crop=None):
    with rawpy.imread('imagesRed/{}.DNG'.format(imgFileName)) as raw:
        imgDim = raw.raw_image.shape

        img = raw.raw_image
        colors = raw.raw_colors

    
    redMask = colors == 0
    #greenMask = np.logical_or((raw.raw_colors == 1), (raw.raw_colors == 3))
    greenMask = colors == 1
    blueMask = colors == 2

    if crop is None:
        numbersBB, spectrumBB = autocrop(img, threshold, redMask, greenMask, blueMask)

        numbers = np.copy(img[numbersBB[1]:(numbersBB[1] + numbersBB[3]), numbersBB[0]:(numbersBB[0] + numbersBB[2])])
        spectrum = np.copy(img[spectrumBB[1]:(spectrumBB[1] + spectrumBB[3]), spectrumBB[0]:(spectrumBB[0] + spectrumBB[2])])

        numbersRedMask = redMask[numbersBB[1]:(numbersBB[1] + numbersBB[3]), numbersBB[0]:(numbersBB[0] + numbersBB[2])]

        redMask = redMask[spectrumBB[1]:(spectrumBB[1] + spectrumBB[3]), spectrumBB[0]:(spectrumBB[0] + spectrumBB[2])]
        greenMask = greenMask[spectrumBB[1]:(spectrumBB[1] + spectrumBB[3]), spectrumBB[0]:(spectrumBB[0] + spectrumBB[2])]
        blueMask = blueMask[spectrumBB[1]:(spectrumBB[1] + spectrumBB[3]), spectrumBB[0]:(spectrumBB[0] + spectrumBB[2])]

    else:

        print('crop :: {}'.format(crop))
        print('dim :: {}'.format(imgDim))
        height, width = imgDim

        numbersBB = np.round(np.array(crop[0:2]) * np.array([height, width])).astype('int32')
        print('Numbers BB :: {}'.format(numbersBB))

        spectrumY = np.round(numbersBB[0, 0] + numbersBB[1, 0])
        print('Spectrum Y :: {}'.format(spectrumY))

        spectrumHeight = np.round(numbersBB[1, 0] * crop[2])
        print('Spectrum Height :: {}'.format(spectrumHeight))

        spectrumBB = np.array([[spectrumY, numbersBB[0, 1]], [spectrumHeight, numbersBB[1][1]]]).astype('int32')
        print('Spectrum BB :: {}'.format(spectrumBB))

        numbersBB = numbersBB.flatten()
        spectrumBB = spectrumBB.flatten()

        numbers = np.copy(img[numbersBB[0]:(numbersBB[0] + numbersBB[2]), numbersBB[1]:(numbersBB[1] + numbersBB[3])])
        spectrum = np.copy(img[spectrumBB[0]:(spectrumBB[0] + spectrumBB[2]), spectrumBB[1]:(spectrumBB[1] + spectrumBB[3])])

        numbersRedMask = redMask[numbersBB[0]:(numbersBB[0] + numbersBB[2]), numbersBB[1]:(numbersBB[1] + numbersBB[3])]

        redMask = redMask[spectrumBB[0]:(spectrumBB[0] + spectrumBB[2]), spectrumBB[1]:(spectrumBB[1] + spectrumBB[3])]
        greenMask = greenMask[spectrumBB[0]:(spectrumBB[0] + spectrumBB[2]), spectrumBB[1]:(spectrumBB[1] + spectrumBB[3])]
        blueMask = blueMask[spectrumBB[0]:(spectrumBB[0] + spectrumBB[2]), spectrumBB[1]:(spectrumBB[1] + spectrumBB[3])]



    #numbersBB, spectrumBB = autocrop(img, threshold, redMask, greenMask, blueMask)
    print('Numbers Size :: {}'.format(numbersBB))
    print('Specturm Size :: {}'.format(spectrumBB))

    numbers = amplify(stretch(numbers))
    #numbers = stretch(numbers)
    spectrum = stretch(spectrum)
    #divide = np.ones([1, numbers.shape[1]])
    #stacked = np.vstack([numbers, divide, spectrum])

    #cv2.imshow('check :: {}'.format(random.random()), stretch(stacked))
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

def spectrumCheck(imgFileName, threshold, crop=None):
    numbers, red, green, blue = extractSpectrums(imgFileName, threshold, crop)

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

    return np.array([quantizedRed, quantizedGreen, quantizedBlue])

#units in terms of 0 to 1... cuz it fits this use case
def scaleReflectanceCurve(reflectance, startWavelength, endWavelength, heightInPixels, startYUnit, endYUnit):
    wavelengthRange = endWavelength - startWavelength
    #yAxisRange = endY - startY

    reflectance[:, 0] -= min(reflectance[:, 0])

    maxReflectance = np.max(reflectance, axis=0)


    scaled = reflectance / maxReflectance
    scaled[:, 1] = (reflectance[:, 1] / heightInPixels) * (endYUnit - startYUnit)
    scaled[:, 0] = startWavelength + (scaled[:, 0] * wavelengthRange)
    scaled[:, 1] = endYUnit - scaled[:, 1]
    quantized = quantizeSpectrum(scaled)

    return quantized

def scaleEmissionCurve(emission, startWavelength, endWavelength):
    wavelengthRange = endWavelength - startWavelength


    emission[:, 0] -= min(emission[:, 0])

    maxEmission = np.max(emission, axis=0)

    scaled = emission / maxEmission
    #scaled[:, 1] = (emission[:, 1] / heightInPixels) * (endYUnit - startYUnit)
    scaled[:, 0] = startWavelength + (scaled[:, 0] * wavelengthRange)
    #scaled[:, 1] = endYUnit - scaled[:, 1]
    quantized = quantizeSpectrum(scaled)

    return quantized

def combineCurves(sourceCurve, targetCurve):
    #for source, target in zip(sourceCurve, targetCurve):
    #    print('{} * {} = {}'.format(source, target, source * target))
    return np.stack([sourceCurve[:, 0], (sourceCurve[:, 1] * targetCurve[:, 1])], axis=1)


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
    combined = np.copy(red)
    combined[:, 1] = np.sum([red[:, 1], green[:, 1], blue[:, 1]], axis=0)

    maxValue = max(combined[:, 1])

    combined[:, 1] /= maxValue

    return combined

def cropSpectrum(spectrum, start, end):
    startIndex = np.argmax(spectrum[:, 0] == start)
    endIndex = np.argmax(spectrum[:, 0] == end)
    return spectrum[startIndex:endIndex, :]

def getCalibrationArray(spectrum, target):
    return target[:, 1] / spectrum[:, 1]

def plotWithScale(red, green, blue, startWavelength, endWavelength):

    scaledRed, scaledGreen, scaledBlue = scalePoints(red, green, blue, startWavelength, endWavelength)

    plt.plot(scaledBlue[:, 0], scaledBlue[:, 1], 'b-')
    plt.plot(scaledGreen[:, 0], scaledGreen[:, 1], 'g-')
    plt.plot(scaledRed[:, 0], scaledRed[:, 1], 'r-')
    plt.show()

def scaleSpectrum(spectrums, wavelengthStart, wavelengthEnd):
    red = spectrums[1][1]
    red = np.stack([np.arange(len(red)), np.array(red)], axis=1)

    green = spectrums[2][1]
    green = np.stack([np.arange(len(green)), np.array(green)], axis=1)

    blue = spectrums[3][1]
    blue = np.stack([np.arange(len(blue)), np.array(blue)], axis=1)

    return scalePoints(red, green, blue, wavelengthStart, wavelengthEnd, False)

def smoothCurve(curve):
    y = savgol_filter(curve[:, 1], 15, 3)
    y /= np.max(y)
    return np.stack([curve[:, 0], y], axis=1)

def calibrateCurve(curve, calibrationCurve):
    y = curve[:, 1] * calibrationCurve
    y /= np.max(y)
    return np.stack([curve[:, 0], y], axis=1)

def spectrumToBGR(spectrum, sensitivity):
    red = spectrum[:, 1] * sensitivity[0][:, 1]
    green = spectrum[:, 1] * sensitivity[1][:, 1]
    blue = spectrum[:, 1] * sensitivity[2][:, 1]

    plt.plot(spectrum[:, 0], red, 'r-')
    plt.plot(spectrum[:, 0], green, 'g-')
    plt.plot(spectrum[:, 0], blue, 'b-')
    #plt.show()

    sums = [np.sum(red), np.sum(green), np.sum(blue)]
    largest = max(sums)
    scaledSums = sums / largest

    scaledSums8bit = [np.round(channel * 255).astype('uint8') for channel in scaledSums]

    #print('BGR :: ({}, {}, {})'.format(blue, green, red))
    print('----------')
    print('RGB :: ({}, {}, {})'.format(*sums))
    print('Scaled RGB :: ({}, {}, {})'.format(*scaledSums))
    print('Scaled RGB 8bit :: ({}, {}, {})'.format(*scaledSums8bit))
    print('----------')
    return [blue, green, red]


cropLED = [[0.470, 0.641], [0.025, 0.162], 1]
cropIncandecentA = [[0.477, 0.649], [0.025, 0.162], 1]
cropIncandecentB = [[0.485, 0.653], [0.025, 0.162], 1]
#crops = [[0.6, 0.6], [0.2, 0.05], 1]
#spectrumCheck(incadecentAImg, 0.08, cropIncandecentA)
#spectrumCheck(incadecentBImg, 0.08, cropIncandecentB)
#spectrumCheck(ledImg, 0.08, cropLED)
#spectrumCheck(iPadImg, 0.30)
#spectrumCheck(benQImg, 0.10)
#cv2.imshow('Force Pause', np.ones([100, 100, 1], dtype='uint8'))
#cv2.waitKey(0)

iPad = extractSpectrums(iPadImg, 0.30)
scaledIpadLightEmission = scaleSpectrum(iPad, 385, 725)

sunlight = extractSpectrums(sunDimImg, 0.08)
scaledSunlightEmission = scaleSpectrum(sunlight, 385, 725)

sky = extractSpectrums(skyImg, 0.50)
scaledSkyLightEmission = scaleSpectrum(sky, 385, 725)

benQ = extractSpectrums(benQImg, 0.10)
scaledBenQLightEmission = scaleSpectrum(benQ, 385, 725)

led = extractSpectrums(ledImg, 0.08, cropLED)
scaledLEDLightEmission = scaleSpectrum(led, 385, 725)

incA = extractSpectrums(incadecentAImg, 0.08, cropIncandecentA)
scaledIncALightEmission = scaleSpectrum(incA, 385, 725)

incB = extractSpectrums(incadecentBImg, 0.08, cropIncandecentB)
scaledIncBLightEmission = scaleSpectrum(incB, 385, 725)

scaledSunlightEmissionGroundTruth = scaleEmissionCurve(sunlightSpectrum, 350, 745)
scaledEyeSensitivity = scalePoints(redEyeCurve, greenEyeCurve, blueEyeCurve, 390, 700, False)

scaledEurope1 = scaleReflectanceCurve(europe1, 400, 700, 1211, 0, 0.55)
scaledEurope2 = scaleReflectanceCurve(europe2, 400, 700, 1211, 0, 0.55)
scaledEurope3 = scaleReflectanceCurve(europe3, 400, 700, 1211, 0, 0.55)

scaledEastAsia1 = scaleReflectanceCurve(eastAsia1, 400, 700, 1211, 0, 0.55)
scaledEastAsia2 = scaleReflectanceCurve(eastAsia2, 400, 700, 1211, 0, 0.55)
scaledEastAsia3 = scaleReflectanceCurve(eastAsia3, 400, 700, 1211, 0, 0.55)

scaledSouthAsia1 = scaleReflectanceCurve(southAsia1, 400, 700, 1211, 0, 0.55)
scaledSouthAsia2 = scaleReflectanceCurve(southAsia2, 400, 700, 1211, 0, 0.55)
scaledSouthAsia3 = scaleReflectanceCurve(southAsia3, 400, 700, 1211, 0, 0.55)

scaledAfrica1 = scaleReflectanceCurve(africa1, 400, 700, 1211, 0, 0.55)
scaledAfrica2 = scaleReflectanceCurve(africa2, 400, 700, 1211, 0, 0.55)
scaledAfrica3 = scaleReflectanceCurve(africa3, 400, 700, 1211, 0, 0.55)
#print('Scaled Europe 1 :: {}'.format(scaledEurope1))

#plt.plot(scaledSunlightEmission[0, :, 0], scaledSunlightEmission[0, :, 1], 'r-')
#plt.plot(scaledSunlightEmission[1, :, 0], scaledSunlightEmission[1, :, 1], 'g-')
#plt.plot(scaledSunlightEmission[2, :, 0], scaledSunlightEmission[2, :, 1], 'b-')
#
#plt.plot(scaledIpadLightEmission[0][:, 0], scaledIpadLightEmission[0][:, 1], 'r--')
#plt.plot(scaledIpadLightEmission[1][:, 0], scaledIpadLightEmission[1][:, 1], 'g--')
#plt.plot(scaledIpadLightEmission[2][:, 0], scaledIpadLightEmission[2][:, 1], 'b--')

#plt.plot(scaledEyeSensitivity[0][:, 0], scaledEyeSensitivity[0][:, 1], 'r.')
#plt.plot(scaledEyeSensitivity[1][:, 0], scaledEyeSensitivity[1][:, 1], 'g.')
#plt.plot(scaledEyeSensitivity[2][:, 0], scaledEyeSensitivity[2][:, 1], 'b.')

#plt.plot(scaledEurope1[:, 0], scaledEurope1[:, 1], 'b-')
#plt.plot(scaledEurope2[:, 0], scaledEurope2[:, 1], 'g-')
#plt.plot(scaledEurope3[:, 0], scaledEurope3[:, 1], 'r-')
#
#plt.plot(scaledEastAsia1[:, 0], scaledEastAsia1[:, 1], 'b--')
#plt.plot(scaledEastAsia2[:, 0], scaledEastAsia2[:, 1], 'b-')
#plt.plot(scaledEastAsia3[:, 0], scaledEastAsia3[:, 1], 'b-')
#
#plt.plot(scaledSouthAsia1[:, 0], scaledSouthAsia1[:, 1], 'g--')
#plt.plot(scaledSouthAsia2[:, 0], scaledSouthAsia2[:, 1], 'g-')
#plt.plot(scaledSouthAsia3[:, 0], scaledSouthAsia3[:, 1], 'g-')
#
#plt.plot(scaledAfrica1[:, 0], scaledAfrica1[:, 1], 'r--')
#plt.plot(scaledAfrica2[:, 0], scaledAfrica2[:, 1], 'r-')
#plt.plot(scaledAfrica3[:, 0], scaledAfrica3[:, 1], 'r-')

startCrop = 420
endCrop = 650

#iphoneWavelengthSensitivityRed = smoothCurve(scaledSunlightEmission[0])
#iphoneWavelengthSensitivityRed = cropSpectrum(iphoneWavelengthSensitivityRed, startCrop, endCrop)
#
#iphoneWavelengthSensitivityGreen = smoothCurve(scaledSunlightEmission[1])
#iphoneWavelengthSensitivityGreen = cropSpectrum(iphoneWavelengthSensitivityGreen, startCrop, endCrop)
#
#iphoneWavelengthSensitivityBlue = smoothCurve(scaledSunlightEmission[2])
#iphoneWavelengthSensitivityBlue = cropSpectrum(iphoneWavelengthSensitivityBlue, startCrop, endCrop)

iphoneWavelengthSensitivityPerChannel = [cropSpectrum(smoothCurve(channel), startCrop, endCrop) for channel in scaledSunlightEmission]
#print('sensitivity per channel :: {}'.format(iphoneWavelengthSensitivityPerChannel))

#plt.plot(iphoneWavelengthSensitivityPerChannel[0][:, 0], iphoneWavelengthSensitivityPerChannel[0][:, 1], 'r-')
#plt.plot(iphoneWavelengthSensitivityPerChannel[1][:, 0], iphoneWavelengthSensitivityPerChannel[1][:, 1], 'g-')
#plt.plot(iphoneWavelengthSensitivityPerChannel[2][:, 0], iphoneWavelengthSensitivityPerChannel[2][:, 1], 'b-')
#plt.show()


iphoneWavelengthSensitivity = combineRGBtoFullSpectrum(*scaledSunlightEmission)
iphoneWavelengthSensitivity = smoothCurve(iphoneWavelengthSensitivity)
iphoneWavelengthSensitivity = cropSpectrum(iphoneWavelengthSensitivity, startCrop, endCrop)
#plt.plot(iphoneWavelengthSensitivity[:, 0], iphoneWavelengthSensitivity[:, 1], 'k-')

scaledSunlightEmissionGroundTruthCroppped = cropSpectrum(scaledSunlightEmissionGroundTruth, startCrop, endCrop)
#plt.plot(scaledSunlightEmissionGroundTruthCroppped[:, 0], scaledSunlightEmissionGroundTruthCroppped[:, 1], 'y-')

calibrationArray = getCalibrationArray(iphoneWavelengthSensitivity, scaledSunlightEmissionGroundTruthCroppped)

ipadWavelengthEmission = combineRGBtoFullSpectrum(*scaledIpadLightEmission)
ipadWavelengthEmission = smoothCurve(ipadWavelengthEmission)
ipadWavelengthEmission = cropSpectrum(ipadWavelengthEmission, startCrop, endCrop)
ipadWavelengthEmissionCalibrated = calibrateCurve(ipadWavelengthEmission, calibrationArray)
#plt.plot(ipadWavelengthEmissionCalibrated[:, 0], ipadWavelengthEmissionCalibrated[:, 1], 'k-')
#plt.plot(ipadWavelengthEmission[:, 0], ipadWavelengthEmission[:, 1], 'b-')

#skyWavelengthEmission = combineRGBtoFullSpectrum(*scaledSkyLightEmission)
#skyWavelengthEmission = cropSpectrum(skyWavelengthEmission, 420, 650)
#plt.plot(skyWavelengthEmission[:, 0], skyWavelengthEmission[:, 1], 'r-')

benqWavelengthEmission = combineRGBtoFullSpectrum(*scaledBenQLightEmission)
benqWavelengthEmission = smoothCurve(benqWavelengthEmission)
benqWavelengthEmission = cropSpectrum(benqWavelengthEmission, startCrop, endCrop)
benqWavelengthEmissionCalibrated = calibrateCurve(benqWavelengthEmission, calibrationArray)
#plt.plot(benqWavelengthEmissionCalibrated[:, 0], benqWavelengthEmissionCalibrated[:, 1], 'b-')

ledWavelengthEmission = combineRGBtoFullSpectrum(*scaledLEDLightEmission)
ledWavelengthEmission = smoothCurve(ledWavelengthEmission)
ledWavelengthEmission = cropSpectrum(ledWavelengthEmission, startCrop, endCrop)
ledWavelengthEmissionCalibrated = calibrateCurve(ledWavelengthEmission, calibrationArray)
#plt.plot(ledWavelengthEmissionCalibrated[:, 0], ledWavelengthEmissionCalibrated[:, 1], 'r-')

incAWavelengthEmission = combineRGBtoFullSpectrum(*scaledIncALightEmission)
incAWavelengthEmission = smoothCurve(incAWavelengthEmission)
incAWavelengthEmission = cropSpectrum(incAWavelengthEmission, startCrop, endCrop)
incAWavelengthEmissionCalibrated = calibrateCurve(incAWavelengthEmission, calibrationArray)
#plt.plot(incAWavelengthEmissionCalibrated[:, 0], incAWavelengthEmissionCalibrated[:, 1], 'g-')


incBWavelengthEmission = combineRGBtoFullSpectrum(*scaledIncBLightEmission)
incBWavelengthEmission = smoothCurve(incBWavelengthEmission)
incBWavelengthEmission = cropSpectrum(incBWavelengthEmission, startCrop, endCrop)
incBWavelengthEmissionCalibrated = calibrateCurve(incBWavelengthEmission, calibrationArray)
#plt.plot(incBWavelengthEmissionCalibrated[:, 0], incBWavelengthEmissionCalibrated[:, 1], 'b-')
#plt.show()

#plt.plot(ipadWavelengthEmission[:, 0], ipadWavelengthEmission[:, 1] * calibrationArray, 'b--')
#plt.plot(benqWavelengthEmission[:, 0], benqWavelengthEmission[:, 1] *calibrationArray, 'b--')

#plt.plot(ledWavelengthEmission[:, 0], ledWavelengthEmission[:, 1] * calibrationArray, 'r--')
#plt.plot(incAWavelengthEmission[:, 0], incAWavelengthEmission[:, 1] * calibrationArray, 'g--')
#plt.plot(incBWavelengthEmission[:, 0], incBWavelengthEmission[:, 1] * calibrationArray, 'b--')

#plt.plot(skyWavelengthEmission[:, 0], skyWavelengthEmission[:, 1] * calibrationArray, 'r--')

scaledEurope1Cropped = cropSpectrum(scaledEurope1, startCrop, endCrop)
#plt.plot(scaledEurope1Cropped[:, 0], scaledEurope1Cropped[:, 1], 'b--')

scaledEurope2Cropped = cropSpectrum(scaledEurope2, startCrop, endCrop)
#plt.plot(scaledEurope2Cropped[:, 0], scaledEurope2Cropped[:, 1], 'g--')

scaledEurope3Cropped = cropSpectrum(scaledEurope3, startCrop, endCrop)
#plt.plot(scaledEurope3Cropped[:, 0], scaledEurope3Cropped[:, 1], 'r--')


scaledEastAsia1Cropped = cropSpectrum(scaledEastAsia1, startCrop, endCrop)
#plt.plot(scaledEastAsia1Cropped[:, 0], scaledEastAsia1Cropped[:, 1], 'b--')

scaledEastAsia2Cropped = cropSpectrum(scaledEastAsia2, startCrop, endCrop)
#plt.plot(scaledEastAsia2Cropped[:, 0], scaledEastAsia2Cropped[:, 1], 'g--')

scaledEastAsia3Cropped = cropSpectrum(scaledEastAsia3, startCrop, endCrop)
#plt.plot(scaledEastAsia3Cropped[:, 0], scaledEastAsia3Cropped[:, 1], 'r--')


scaledSouthAsia1Cropped = cropSpectrum(scaledSouthAsia1, startCrop, endCrop)
#plt.plot(scaledSouthAsia1Cropped[:, 0], scaledSouthAsia1Cropped[:, 1], 'b--')

scaledSouthAsia2Cropped = cropSpectrum(scaledSouthAsia2, startCrop, endCrop)
#plt.plot(scaledSouthAsia2Cropped[:, 0], scaledSouthAsia2Cropped[:, 1], 'g--')

scaledSouthAsia3Cropped = cropSpectrum(scaledSouthAsia3, startCrop, endCrop)
#plt.plot(scaledSouthAsia3Cropped[:, 0], scaledSouthAsia3Cropped[:, 1], 'r--')


scaledAfrica1Cropped = cropSpectrum(scaledAfrica1, startCrop, endCrop)
#plt.plot(scaledAfrica1Cropped[:, 0], scaledAfrica1Cropped[:, 1], 'b--')

scaledAfrica2Cropped = cropSpectrum(scaledAfrica2, startCrop, endCrop)
#plt.plot(scaledAfrica2Cropped[:, 0], scaledAfrica2Cropped[:, 1], 'g--')

scaledAfrica3Cropped = cropSpectrum(scaledAfrica3, startCrop, endCrop)
#plt.plot(scaledAfrica3Cropped[:, 0], scaledAfrica3Cropped[:, 1], 'r--')



combinedEurope1Ipad = combineCurves(scaledEurope1Cropped, ipadWavelengthEmissionCalibrated)
plt.plot(combinedEurope1Ipad[:, 0], combinedEurope1Ipad[:, 1], 'b-')

combinedEurope1Benq = combineCurves(scaledEurope1Cropped, benqWavelengthEmissionCalibrated)
combinedEurope1Led = combineCurves(scaledEurope1Cropped, ledWavelengthEmissionCalibrated)
combinedEurope1IncA = combineCurves(scaledEurope1Cropped, incAWavelengthEmissionCalibrated)
combinedEurope1IncB = combineCurves(scaledEurope1Cropped, incBWavelengthEmissionCalibrated)

combinedEurope2Ipad = combineCurves(scaledEurope2Cropped, ipadWavelengthEmissionCalibrated)
plt.plot(combinedEurope2Ipad[:, 0], combinedEurope2Ipad[:, 1], 'g-')

combinedEurope3Ipad = combineCurves(scaledEurope3Cropped, ipadWavelengthEmissionCalibrated)
plt.plot(combinedEurope3Ipad[:, 0], combinedEurope3Ipad[:, 1], 'r-')


combinedEastAsia1Ipad = combineCurves(scaledEastAsia1Cropped, ipadWavelengthEmissionCalibrated)
plt.plot(combinedEastAsia1Ipad[:, 0], combinedEastAsia1Ipad[:, 1], 'b-')

combinedEastAsia2Ipad = combineCurves(scaledEastAsia2Cropped, ipadWavelengthEmissionCalibrated)
plt.plot(combinedEastAsia2Ipad[:, 0], combinedEastAsia2Ipad[:, 1], 'g-')

combinedEastAsia3Ipad = combineCurves(scaledEastAsia3Cropped, ipadWavelengthEmissionCalibrated)
plt.plot(combinedEastAsia3Ipad[:, 0], combinedEastAsia3Ipad[:, 1], 'r-')


combinedSouthAsia1Ipad = combineCurves(scaledSouthAsia1Cropped, ipadWavelengthEmissionCalibrated)
plt.plot(combinedSouthAsia1Ipad[:, 0], combinedSouthAsia1Ipad[:, 1], 'b-')

combinedSouthAsia2Ipad = combineCurves(scaledSouthAsia2Cropped, ipadWavelengthEmissionCalibrated)
plt.plot(combinedSouthAsia2Ipad[:, 0], combinedSouthAsia2Ipad[:, 1], 'g-')

combinedSouthAsia3Ipad = combineCurves(scaledSouthAsia3Cropped, ipadWavelengthEmissionCalibrated)
plt.plot(combinedSouthAsia3Ipad[:, 0], combinedSouthAsia3Ipad[:, 1], 'r-')


combinedAfrica1Ipad = combineCurves(scaledAfrica1Cropped, ipadWavelengthEmissionCalibrated)
plt.plot(combinedAfrica1Ipad[:, 0], combinedAfrica1Ipad[:, 1], 'b-')

combinedAfrica2Ipad = combineCurves(scaledAfrica2Cropped, ipadWavelengthEmissionCalibrated)
plt.plot(combinedAfrica2Ipad[:, 0], combinedAfrica2Ipad[:, 1], 'g-')

combinedAfrica3Ipad = combineCurves(scaledAfrica3Cropped, ipadWavelengthEmissionCalibrated)
plt.plot(combinedAfrica3Ipad[:, 0], combinedAfrica3Ipad[:, 1], 'r-')


combinedEurope1Sunlight = combineCurves(scaledEurope1Cropped, scaledSunlightEmissionGroundTruthCroppped)
combinedEurope2Sunlight = combineCurves(scaledEurope2Cropped, scaledSunlightEmissionGroundTruthCroppped)
combinedEurope3Sunlight = combineCurves(scaledEurope3Cropped, scaledSunlightEmissionGroundTruthCroppped)
#plt.plot(combinedEurope1Sunlight[:, 0], combinedEurope1Sunlight[:, 1], 'g-')

plt.show()

#print('Scaled :: {}'.format(scaledSunlightEmission))

print('IPAD')
print('-Europe 1')
spectrumToBGR(combinedEurope1Ipad, iphoneWavelengthSensitivityPerChannel)
print('-Europe 2')
spectrumToBGR(combinedEurope2Ipad, iphoneWavelengthSensitivityPerChannel)
print('-Europe 3')
spectrumToBGR(combinedEurope3Ipad, iphoneWavelengthSensitivityPerChannel)
print('SUNLIGHT')
print('-Europe 1')
spectrumToBGR(combinedEurope1Sunlight, iphoneWavelengthSensitivityPerChannel)
print('-Europe 2')
spectrumToBGR(combinedEurope2Sunlight, iphoneWavelengthSensitivityPerChannel)
print('-Europe 3')
spectrumToBGR(combinedEurope3Sunlight, iphoneWavelengthSensitivityPerChannel)
#print('SUNLIGHT')
#spectrumToBGR(combinedEurope1Sunlight, iphoneWavelengthSensitivityPerChannel)
#print('BenQ')
#spectrumToBGR(combinedEurope1Benq, iphoneWavelengthSensitivityPerChannel)
#print('LED')
#spectrumToBGR(combinedEurope1Led, iphoneWavelengthSensitivityPerChannel)
#print('IncadecentA')
#spectrumToBGR(combinedEurope1IncA, iphoneWavelengthSensitivityPerChannel)
#print('IncadecentB')
#spectrumToBGR(combinedEurope1IncB, iphoneWavelengthSensitivityPerChannel)
plt.show()

#spectrumCheck(sunDimImg, 0.08)
#spectrumCheck(skyImg, 0.5)
#spectrumCheck(sunMediumImg, 0.50)
#spectrumCheck(benQImg, 0.10)
#spectrumCheck(iPadImg, 0.30)
#spectrumCheck(incadecentAImg, 0.01)
#spectrumCheck(incadecentBImg, 0.10)
#spectrumCheck(ledImg, 0.10)
cv2.imshow('Force Pause', np.ones([100, 100, 1], dtype='uint8'))
cv2.waitKey(0)
