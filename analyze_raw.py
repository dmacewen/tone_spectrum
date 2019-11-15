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
sunMediumImg = 'sunlightMedium'#395nm -> 700nm
#sunDimImg = 'sunlightDim'
benQImg = 'BenQ2'
iPadImg = 'iPad' #400nm -> 710nm

incadecentAImg = 'IncadecentA'
incadecentBImg = 'IncadecentB'
incadecentBBrightImg = 'IncadecentB_bright'
ledImg = 'LED' 

blueEyeCurve = np.array([[0,3],[9,10],[12,18],[24,91],[25,115],[39,181],[42,189],[43,189],[45,180],[52,157],[55,152],[58,131],[61,125],[77,41],[87,20],[92,16],[95,10],[109,2],[114,1],[120,0],[256,0]])
greenEyeCurve = np.array([[0,0],[15,3],[22,5],[27,7],[48,16],[55,21],[66,41],[76,56],[84,74],[91,108],[98,147],[108,181],[114,188],[116,189],[125,188],[127,186],[128,182],[138,158],[140,157],[157,96],[159,80],[168,50],[173,35],[185,14],[192,7],[210,1],[211,0],[256,0]])
redEyeCurve = np.array([[0,0],[25,3],[45,8],[58,14],[74,30],[82,45],[87,60],[95,92],[103,130],[109,150],[114,159],[124,177],[128,183],[131,183],[134,184],[139,188],[143,189],[145,188],[148,185],[149,183],[152,182],[157,175],[168,148],[179,110],[192,63],[201,35],[205,25],[213,15],[222,7],[230,3],[240,0], [256,0]])

sunlightSpectrum = np.array([[0, 0], [55, 0], [112, 3], [167, 12], [222, 14], [279, 21], [334, 51], [390, 130], [446, 271], [501, 441], [557, 543], [613, 581], [668, 617], [724, 640], [780, 655], [835, 774], [891, 993], [947, 1176], [1003, 1295], [1058, 1394], [1113, 1501], [1170, 1573], [1225, 1594], [1280, 1574], [1337, 1513], [1392, 1459], [1448, 1501], [1504, 1609], [1559, 1722], [1615, 1857], [1671, 1991], [1727, 2107], [1782, 2194], [1838, 2259], [1894, 2320], [1949, 2356], [2006, 2338], [2061, 2305], [2116, 2318], [2172, 2346], [2228, 2365], [2283, 2356], [2339, 2295], [2395, 2293], [2450, 2383], [2506, 2461], [2562, 2472], [2618, 2474], [2673, 2503], [2729, 2513], [2785, 2503], [2840, 2457], [2897, 2420], [2952, 2460], [3007, 2515], [3064, 2566], [3119, 2594], [3174, 2586], [3230, 2555], [3286, 2553], [3342, 2591], [3397, 2603], [3453, 2585], [3509, 2572], [3564, 2600], [3620, 2646], [3676, 2641], [3731, 2614], [3788, 2634], [3843, 2672], [3898, 2689], [3955, 2700], [4010, 2675], [4065, 2609], [4122, 2591], [4177, 2631], [4233, 2648], [4288, 2666], [4344, 2653], [4400, 2606], [4455, 2569], [4511, 2582], [4567, 2616], [4622, 2606], [4679, 2560], [4734, 2544], [4789, 2574], [4846, 2619], [4901, 2627], [4956, 2612], [5013, 2599], [5068, 2601], [5124, 2590], [5180, 2559], [5235, 2577], [5291, 2632], [5346, 2676], [5402, 2702], [5458, 2709], [5513, 2685], [5570, 2637], [5625, 2555], [5680, 2455], [5737, 2395], [5792, 2428], [5847, 2450], [5904, 2425], [5959, 2415], [6015, 2470], [6071, 2530], [6126, 2520], [6182, 2402], [6238, 2238], [6293, 2197], [6349, 2258], [6404, 2288], [6461, 2257], [6516, 2164], [6572, 1902]])

europe1 = np.array([
     [0, 428],
     [75, 466],
     [104, 478],
     [140, 488],
     [154, 488],
     [175, 485],
     [204, 471],
     [244, 432],
     [311, 368],
     [354, 337],
     [386, 323],
     [446, 308],
     [469, 306],
     [519, 302],
     [632, 285],
     [664, 283],
     [739, 284],
     [764, 287],
     [814, 304],
     [873, 324],
     [911, 334],
     [989, 340],
     [1059, 342],
     [1132, 354],
     [1164, 352],
     [1195, 339],
     [1207, 326],
     [1251, 289],
     [1258, 282],
     [1314, 252],
     [1378, 234],
     [1405, 234],
     [1560, 242],
     [1797, 262],
     [1829, 264],
     [1946, 270]])
europe2 = np.array([
    [0, 499],
    [19, 513],
    [58, 532],
    [116, 551],
    [141, 554],
    [169, 549],
    [201, 530],
    [217, 511],
    [254, 464],
    [293, 392],
    [324, 344],
    [364, 306],
    [414, 276],
    [464, 256],
    [563, 234],
    [709, 218],
    [764, 222],
    [804, 236],
    [866, 268],
    [904, 285],
    [930, 292],
    [1106, 294],
    [1150, 293],
    [1169, 286],
    [1195, 266],
    [1222, 234],
    [1239, 208],
    [1274, 171],
    [1311, 139],
    [1344, 120],
    [1374, 113],
    [1417, 104],
    [1505, 100],
    [1605, 103],
    [1838, 119],
    [1885, 120],
    [1946, 123]])
europe3 = np.array([
    [0, 593],
    [15, 600],
    [51, 616],
    [63, 621],
    [115, 637],
    [137, 640],
    [176, 640],
    [214, 620],
    [245, 588],
    [332, 460],
    [347, 442],
    [386, 417],
    [416, 405],
    [484, 389],
    [570, 377],
    [590, 373],
    [647, 366],
    [694, 366],
    [769, 371],
    [784, 375],
    [831, 396],
    [899, 417],
    [997, 425],
    [1106, 431],
    [1164, 422],
    [1204, 396],
    [1239, 362],
    [1279, 329],
    [1325, 302],
    [1374, 285],
    [1414, 284],
    [1471, 285],
    [1490, 285],
    [1520, 286],
    [1714, 299],
    [1769, 301],
    [1946, 312]])
eastAsia1 = np.array([
    [0, 663],
    [22, 662],
    [53, 665],
    [71, 668],
    [116, 677],
    [159, 671],
    [206, 652],
    [232, 633],
    [331, 511],
    [389, 470],
    [439, 452],
    [476, 445],
    [583, 423],
    [636, 402],
    [691, 392],
    [764, 390],
    [832, 404],
    [906, 426],
    [964, 435],
    [1106, 433],
    [1164, 422],
    [1204, 396],
    [1239, 360],
    [1279, 327],
    [1325, 301],
    [1407, 279],
    [1456, 271],
    [1584, 272],
    [1694, 278],
    [1788, 280],
    [1893, 281],
    [1946, 281]])
eastAsia2 = np.array([
    [0, 692],
    [11, 690],
    [24, 683],
    [70, 663],
    [114, 638],
    [157, 629],
    [186, 614],
    [251, 555],
    [333, 439],
    [369, 407],
    [404, 384],
    [464, 359],
    [542, 337],
    [681, 303],
    [749, 296],
    [804, 305],
    [894, 337],
    [944, 342],
    [1018, 339],
    [1127, 334],
    [1161, 328],
    [1194, 304],
    [1243, 256],
    [1299, 212],
    [1349, 191],
    [1412, 180],
    [1464, 178],
    [1586, 182],
    [1838, 206],
    [1946, 221]])
eastAsia3 = np.array([
    [0, 1048],
    [53, 996],
    [84, 969],
    [167, 916],
    [238, 864],
    [310, 788],
    [414, 720],
    [595, 657],
    [670, 636],
    [734, 624],
    [859, 617],
    [1064, 598],
    [1154, 579],
    [1184, 567],
    [1215, 548],
    [1286, 493],
    [1374, 459],
    [1476, 446],
    [1829, 414],
    [1946, 410]])
southAsia1 =np.array([
    [0, 758],
    [128, 735],
    [175, 718],
    [249, 664],
    [326, 574],
    [404, 515],
    [467, 489],
    [608, 454],
    [705, 424],
    [793, 417],
    [882, 419],
    [976, 421],
    [1104, 404],
    [1144, 394],
    [1179, 378],
    [1211, 359],
    [1294, 285],
    [1315, 273],
    [1364, 254],
    [1464, 229],
    [1586, 214],
    [1743, 204],
    [1832, 196],
    [1946, 192]])
southAsia2 =np.array([
    [0, 850],
    [92, 831],
    [160, 822],
    [214, 798],
    [271, 752],
    [294, 724],
    [344, 684],
    [384, 654],
    [433, 634],
    [494, 614],
    [572, 598],
    [664, 579],
    [714, 569],
    [843, 564],
    [963, 564],
    [1047, 559],
    [1104, 549],
    [1204, 512],
    [1244, 482],
    [1311, 442],
    [1371, 412],
    [1419, 397],
    [1649, 382],
    [1946, 365]])
southAsia3 =np.array([
    [0, 1010],
    [54, 1014],
    [103, 1009],
    [170, 993],
    [324, 924],
    [374, 899],
    [504, 862],
    [701, 808],
    [876, 777],
    [1048, 754],
    [1164, 719],
    [1232, 679],
    [1334, 609],
    [1516, 553],
    [1575, 538],
    [1747, 502],
    [1946, 464]])
africa1 =np.array([
    [0, 999],
    [121, 986],
    [187, 974],
    [284, 936],
    [384, 887],
    [505, 854],
    [702, 811],
    [841, 790],
    [1054, 764],
    [1140, 742],
    [1240, 694],
    [1323, 644],
    [1417, 604],
    [1620, 546],
    [1795, 505],
    [1946, 479]])
africa2 = np.array([
    [0, 1050],
    [34, 1054],
    [114, 1044],
    [221, 1016],
    [414, 944],
    [700, 877],
    [845, 856],
    [933, 844],
    [1130, 802],
    [1164, 792],
    [1214, 764],
    [1269, 725],
    [1342, 675],
    [1374, 658],
    [1533, 608],
    [1630, 578],
    [1764, 544],
    [1841, 526],
    [1946, 505]])
africa3 = np.array([
    [0, 1115],
    [44, 1096],
    [101, 1078],
    [247, 1044],
    [302, 1022],
    [454, 977],
    [702, 918],
    [846, 895],
    [917, 886],
    [1113, 846],
    [1184, 824],
    [1294, 764],
    [1335, 744],
    [1563, 672],
    [1703, 622],
    [1866, 568],
    [1946, 544]])



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
    for source, target in zip(sourceCurve, targetCurve):
        print('{} * {} = {}'.format(source, target, source * target))


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

#plt.plot(scaledEurope1[:, 0], scaledEurope1[:, 1], 'k--')
#plt.plot(scaledEurope2[:, 0], scaledEurope2[:, 1], 'k-')
#plt.plot(scaledEurope3[:, 0], scaledEurope3[:, 1], 'k-')
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

iphoneWavelengthSensitivity = combineRGBtoFullSpectrum(*scaledSunlightEmission)
iphoneWavelengthSensitivity = smoothCurve(iphoneWavelengthSensitivity)
iphoneWavelengthSensitivity = cropSpectrum(iphoneWavelengthSensitivity, startCrop, endCrop)
#plt.plot(iphoneWavelengthSensitivity[:, 0], iphoneWavelengthSensitivity[:, 1], 'k-')

scaledSunlightCroppped = cropSpectrum(scaledSunlightEmissionGroundTruth, startCrop, endCrop)
#plt.plot(scaledSunlightCroppped[:, 0], scaledSunlightCroppped[:, 1], 'y-')

calibrationArray = getCalibrationArray(iphoneWavelengthSensitivity, scaledSunlightCroppped)

ipadWavelengthEmission = combineRGBtoFullSpectrum(*scaledIpadLightEmission)
ipadWavelengthEmission = smoothCurve(ipadWavelengthEmission)
ipadWavelengthEmission = cropSpectrum(ipadWavelengthEmission, startCrop, endCrop)
ipadWavelengthEmissionCalibrated = calibrateCurve(ipadWavelengthEmission, calibrationArray)
plt.plot(ipadWavelengthEmissionCalibrated[:, 0], ipadWavelengthEmissionCalibrated[:, 1], 'k-')
#plt.plot(ipadWavelengthEmission[:, 0], ipadWavelengthEmission[:, 1], 'b-')

#skyWavelengthEmission = combineRGBtoFullSpectrum(*scaledSkyLightEmission)
#skyWavelengthEmission = cropSpectrum(skyWavelengthEmission, 420, 650)
#plt.plot(skyWavelengthEmission[:, 0], skyWavelengthEmission[:, 1], 'r-')

#benqWavelengthEmission = combineRGBtoFullSpectrum(*scaledBenQLightEmission)
#benqWavelengthEmission = cropSpectrum(benqWavelengthEmission, 420, 650)
#plt.plot(benqWavelengthEmission[:, 0], benqWavelengthEmission[:, 1], 'b-')

ledWavelengthEmission = combineRGBtoFullSpectrum(*scaledLEDLightEmission)
ledWavelengthEmission = smoothCurve(ledWavelengthEmission)
ledWavelengthEmission = cropSpectrum(ledWavelengthEmission, startCrop, endCrop)
#plt.plot(ledWavelengthEmission[:, 0], ledWavelengthEmission[:, 1], 'r-')
ledWavelengthEmissionCalibrated = calibrateCurve(ledWavelengthEmission, calibrationArray)
plt.plot(ledWavelengthEmissionCalibrated[:, 0], ledWavelengthEmissionCalibrated[:, 1], 'r-')

incAWavelengthEmission = combineRGBtoFullSpectrum(*scaledIncALightEmission)
incAWavelengthEmission = smoothCurve(incAWavelengthEmission)
incAWavelengthEmission = cropSpectrum(incAWavelengthEmission, startCrop, endCrop)
#plt.plot(incAWavelengthEmission[:, 0], incAWavelengthEmission[:, 1], 'g-')
incAWavelengthEmissionCalibrated = calibrateCurve(incAWavelengthEmission, calibrationArray)
plt.plot(incAWavelengthEmissionCalibrated[:, 0], incAWavelengthEmissionCalibrated[:, 1], 'g-')


incBWavelengthEmission = combineRGBtoFullSpectrum(*scaledIncBLightEmission)
incBWavelengthEmission = smoothCurve(incBWavelengthEmission)
incBWavelengthEmission = cropSpectrum(incBWavelengthEmission, startCrop, endCrop)
#plt.plot(incBWavelengthEmission[:, 0], incBWavelengthEmission[:, 1], 'b-')
incBWavelengthEmissionCalibrated = calibrateCurve(incBWavelengthEmission, calibrationArray)
plt.plot(incBWavelengthEmissionCalibrated[:, 0], incBWavelengthEmissionCalibrated[:, 1], 'b-')

#plt.plot(ipadWavelengthEmission[:, 0], ipadWavelengthEmission[:, 1] * calibrationArray, 'b--')
#plt.plot(benqWavelengthEmission[:, 0], benqWavelengthEmission[:, 1] *calibrationArray, 'b--')

#plt.plot(ledWavelengthEmission[:, 0], ledWavelengthEmission[:, 1] * calibrationArray, 'r--')
#plt.plot(incAWavelengthEmission[:, 0], incAWavelengthEmission[:, 1] * calibrationArray, 'g--')
#plt.plot(incBWavelengthEmission[:, 0], incBWavelengthEmission[:, 1] * calibrationArray, 'b--')

#plt.plot(skyWavelengthEmission[:, 0], skyWavelengthEmission[:, 1] * calibrationArray, 'r--')

plt.show()

print('Scaled :: {}'.format(scaledSunlightEmission))

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
