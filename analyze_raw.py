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

europe1 = np.array([
     [238, 584],
     [311, 622],
     [340, 634],
     [376, 644],
     [390, 644],
     [411, 641],
     [440, 627],
     [480, 588],
     [547, 524],
     [590, 493],
     [622, 479],
     [682, 464],
     [705, 462],
     [755, 458],
     [868, 441],
     [900, 439],
     [975, 440],
     [1000, 443],
     [1050, 460],
     [1109, 480],
     [1147, 490],
     [1225, 496],
     [1295, 498],
     [1368, 510],
     [1400, 508],
     [1431, 495],
     [1443, 482],
     [1487, 445],
     [1494, 438],
     [1550, 408],
     [1614, 390],
     [1641, 390],
     [1796, 398],
     [2033, 418],
     [2065, 420],
     [2184, 426]])
europe2 = np.array([
    [238, 655],
    [255, 669],
    [294, 688],
    [352, 707],
    [377, 710],
    [405, 705],
    [437, 686],
    [453, 667],
    [490, 620],
    [529, 548],
    [560, 500],
    [600, 462],
    [650, 432],
    [700, 412],
    [799, 390],
    [945, 374],
    [1000, 378],
    [1040, 392],
    [1102, 424],
    [1140, 441],
    [1166, 448],
    [1342, 450],
    [1386, 449],
    [1405, 442],
    [1431, 422],
    [1458, 390],
    [1475, 364],
    [1510, 327],
    [1547, 295],
    [1580, 276],
    [1610, 269],
    [1653, 260],
    [1741, 256],
    [1841, 259],
    [2074, 275],
    [2121, 276],
    [2181, 279]])
europe3 = np.array([
    [236, 749],
    [251, 756],
    [287, 772],
    [299, 777],
    [351, 793],
    [373, 796],
    [412, 796],
    [450, 776],
    [481, 744],
    [568, 616],
    [583, 598],
    [622, 573],
    [652, 561],
    [720, 545],
    [806, 533],
    [826, 529],
    [883, 522],
    [930, 522],
    [1005, 527],
    [1020, 531],
    [1067, 552],
    [1135, 573],
    [1233, 581],
    [1342, 587],
    [1400, 578],
    [1440, 552],
    [1475, 518],
    [1515, 485],
    [1561, 458],
    [1610, 441],
    [1650, 440],
    [1707, 441],
    [1726, 441],
    [1756, 442],
    [1950, 455],
    [2005, 457],
    [2180, 468]])
eastAsia1 = np.array([
    [236, 819],
    [258, 818],
    [289, 821],
    [307, 824],
    [352, 833],
    [395, 827],
    [442, 808],
    [468, 789],
    [567, 667],
    [625, 626],
    [675, 608],
    [712, 601],
    [819, 579],
    [872, 558],
    [927, 548],
    [1000, 546],
    [1068, 560],
    [1142, 582],
    [1200, 591],
    [1342, 589],
    [1400, 578],
    [1440, 552],
    [1475, 516],
    [1515, 483],
    [1561, 457],
    [1643, 435],
    [1692, 427],
    [1820, 428],
    [1930, 434],
    [2024, 436],
    [2129, 437],
    [2182, 437]])
eastAsia2 = np.array([
    [236, 848],
    [247, 846],
    [260, 839],
    [306, 819],
    [306, 819],
    [350, 794],
    [393, 785],
    [422, 770],
    [487, 711],
    [569, 595],
    [605, 563],
    [640, 540],
    [700, 515],
    [778, 493],
    [917, 459],
    [985, 452],
    [1040, 461],
    [1130, 493],
    [1180, 498],
    [1254, 495],
    [1363, 490],
    [1397, 484],
    [1430, 460],
    [1479, 412],
    [1535, 368],
    [1585, 347],
    [1648, 336],
    [1700, 334],
    [1822, 338],
    [2074, 362],
    [2182, 377]])
eastAsia3 = np.array([
    [236, 1204],
    [289, 1152],
    [320, 1125],
    [403, 1072],
    [474, 1020],
    [546, 944],
    [650, 876],
    [831, 813],
    [906, 792],
    [970, 780],
    [1095, 773],
    [1300, 754],
    [1390, 735],
    [1420, 723],
    [1451, 704],
    [1522, 649],
    [1610, 615],
    [1712, 602],
    [2065, 570],
    [2182, 566]])
southAsia1 =np.array([
    [238, 914],
    [364, 891],
    [411, 874],
    [485, 820],
    [562, 730],
    [640, 671],
    [703, 645],
    [844, 610],
    [941, 580],
    [1029, 573],
    [1118, 575],
    [1212, 577],
    [1340, 560],
    [1380, 550],
    [1415, 534],
    [1447, 515],
    [1530, 441],
    [1551, 429],
    [1600, 410],
    [1700, 385],
    [1822, 370],
    [1979, 360],
    [2068, 352],
    [2182, 348]])
southAsia2 =np.array([
    [236, 1006],
    [328, 987],
    [396, 978],
    [450, 954],
    [507, 908],
    [530, 880],
    [580, 840],
    [620, 810],
    [669, 790],
    [730, 770],
    [808, 754],
    [900, 735],
    [950, 725],
    [1079, 720],
    [1199, 720],
    [1283, 715],
    [1340, 705],
    [1440, 668],
    [1480, 638],
    [1547, 598],
    [1607, 568],
    [1655, 553],
    [1885, 538],
    [2182, 521]])
southAsia3 =np.array([
    [235, 1166],
    [290, 1170],
    [339, 1165],
    [406, 1149],
    [560, 1080],
    [610, 1055],
    [740, 1018],
    [937, 964],
    [1112, 933],
    [1284, 910],
    [1400, 875],
    [1468, 835],
    [1570, 765],
    [1752, 709],
    [1811, 694],
    [1983, 658],
    [2182, 620]])
africa1 =np.array([
    [237, 1155],
    [357, 1142],
    [423, 1130],
    [520, 1092],
    [620, 1043],
    [741, 1010],
    [938, 967],
    [1077, 946],
    [1290, 920],
    [1376, 898],
    [1476, 850],
    [1559, 800],
    [1653, 760],
    [1856, 702],
    [2031, 661],
    [2182, 635]])
africa2 = np.array([
    [236, 1206],
    [270, 1210],
    [350, 1200],
    [457, 1172],
    [650, 1100],
    [936, 1033],
    [1081, 1012],
    [1169, 1000],
    [1366, 958],
    [1400, 948],
    [1450, 920],
    [1505, 881],
    [1578, 831],
    [1610, 814],
    [1769, 764],
    [1866, 734],
    [2000, 700],
    [2077, 682],
    [2182, 661]])
africa3 = np.array([
    [237, 1271],
    [280, 1252],
    [337, 1234],
    [483, 1200],
    [538, 1178],
    [690, 1133],
    [938, 1074],
    [1082, 1051],
    [1153, 1042],
    [1349, 1002],
    [1420, 980],
    [1530, 920],
    [1571, 900],
    [1799, 828],
    [1939, 778],
    [2102, 724],
    [2182, 700]])



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

def scaleReflectanceCurve(reflectance, startWavelength, endWavelength, startY, endY):
    wavelengthRange = endWavelength - startWavelength
    yAxisRange = endY - startY

    reflectance[:, 0] -= min(reflectance[:, 0])

    maxReflectance = np.max(reflectance, axis=0)


    scaled = reflectance / maxReflectance
    scaled[:, 0] = startWavelength + (scaled[:, 0] * wavelengthRange)
    scaled[:, 1] = startY + (scaled[:, 1] * yAxisRange)
    quantized = quantizeSpectrum(scaled)

    return quantized

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

scaledEurope1 = scaleReflectanceCurve(europe1, 400, 700)
print('Scaled Europe 1 :: {}'.format(scaledEurope1))

#plt.plot(scaledSunlightEmission[0][:, 0], scaledSunlightEmission[0][:, 1], 'r-')
#plt.plot(scaledSunlightEmission[1][:, 0], scaledSunlightEmission[1][:, 1], 'g-')
#plt.plot(scaledSunlightEmission[2][:, 0], scaledSunlightEmission[2][:, 1], 'b-')

plt.plot(scaledIpadLightEmission[0][:, 0], scaledIpadLightEmission[0][:, 1], 'r--')
plt.plot(scaledIpadLightEmission[1][:, 0], scaledIpadLightEmission[1][:, 1], 'g--')
plt.plot(scaledIpadLightEmission[2][:, 0], scaledIpadLightEmission[2][:, 1], 'b--')

#plt.plot(scaledEyeSensitivity[0][:, 0], scaledEyeSensitivity[0][:, 1], 'r-')
#plt.plot(scaledEyeSensitivity[1][:, 0], scaledEyeSensitivity[1][:, 1], 'g-')
#plt.plot(scaledEyeSensitivity[2][:, 0], scaledEyeSensitivity[2][:, 1], 'b-')

plt.plot(scaledEurope1[:, 0], scaledEurope1[:, 1], 'k-')
plt.show()

print('Scaled :: {}'.format(scaledSunlightEmission))


#spectrumCheck(sunDimImg, 0.30)
#spectrumCheck(sunMediumImg, 0.50)
#spectrumCheck(benQImg, 0.10)
#spectrumCheck(iPadImg, 0.30)
cv2.imshow('Force Pause', np.ones([100, 100, 1], dtype='uint8'))
cv2.waitKey(0)
