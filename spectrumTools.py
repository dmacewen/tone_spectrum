import csv
import numpy as np
import matplotlib.pyplot as plt
import copy
from scipy.signal import savgol_filter

targetWavelengthRange = [420, 650]

#Curve Object
# - Curve Points
# - startWavelength, endWavelength
# - startYAxisPixel, endYAxisPixel
# - startYAxisRatio, endYAxisRatio

def makeCurveObject(curve, wavelengthRange, yAxisPixelRange, yAxisRatioRange):
    curveObject = {}
    curveObject['curve'] = curve
    curveObject['wavelengthRange'] = wavelengthRange
    curveObject['yAxisPixelRange'] = yAxisPixelRange
    curveObject['yAxisRatioRange'] = yAxisRatioRange

    return curveObject

def readCurve(path):
    with open(path, newline='') as csvFile:
        reader = csv.reader(csvFile, delimiter=',')
        rawFilePairs = [pair for pair in reader]

        wavelengthRange = [int(num) for num in rawFilePairs[0]]
        yAxisPixelRange = [int(num) for num in rawFilePairs[1]]
        yAxisRatioRange = [float(num) for num in rawFilePairs[2]]
        curve = np.asarray([[int(pair[0]), int(pair[1])] for pair in rawFilePairs[3:]])

        return makeCurveObject(curve, wavelengthRange, yAxisPixelRange, yAxisRatioRange)

def readTracedCurves(fileName):
    return readCurve('curves/tracedCurves/{}.csv'.format(fileName))

def readMeasuredCurves(fileName):
    return readCurve('curves/measuredCurves/{}.csv'.format(fileName))

def writeCurve(path, curveObject):
    with open(path, 'w', newline='') as csvFile:
        writer = csv.writer(csvFile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(curveObject['wavelengthRange'])
        writer.writerow(curveObject['yAxisPixelRange'])
        writer.writerow(curveObject['yAxisRatioRange'])
        writer.writerows(curveObject['curve'])

def writeMeasuredCurve(fileName, curveObject):
    writeCurve('curves/measuredCurves/{}.csv'.format(fileName), curveObject)

def writeComputedCurve(fileName, curveObject):
    writeCurve('curves/computedCurves/{}.csv'.format(fileName), curveObject)

#y=mx+b
#m = rise/run... (y2 - y1) / (x2 - x1)
#b = y1 - m * x1
def quantizeCurve(curveObject):
    curve = curveObject['curve']

    #sortedPoints = np.array(sorted(curve, key=lambda point: point[0]))
    pointDiffs = curve[1:] - curve[:-1]

    slopes = pointDiffs[:, 1] / pointDiffs[:, 0]
    intercepts = curve[:-1, 1] - (slopes * curve[:-1, 0])
    slopeRanges = np.stack([curve[:-1, 0], curve[1:, 0]], axis=1)

    points = []
    for m, b, lineRange in zip(slopes, intercepts, slopeRanges):
        if (lineRange[0] == lineRange[1]) or (lineRange[0] == (lineRange[1] + 1)):
            xValues = lineRange[0]
        else:
            xValues = np.arange(np.floor(lineRange[0]), np.floor(lineRange[1]))
        yValues = xValues * m + b
        points.append(np.stack([xValues, yValues], axis=1))

    curve = np.concatenate(points, axis=0)
    curveObject['curve'] = curve

    return curveObject

#Smooth after scaling! 
def smoothCurve(curveObject):
    curve = curveObject['curve']
    y = savgol_filter(curve[:, 1], 15, 3)
    curveObject['curve'] = np.stack([curve[:, 0], y], axis=1)
    return curveObject

def scaleCurve(curveObject):
    curve = curveObject['curve']
    startWavelength, endWavelength = curveObject['wavelengthRange']
    wavelengthRange = endWavelength - startWavelength

    startYAxisPixel, endYAxisPixel = curveObject['yAxisPixelRange']
    yAxisPixelRange = endYAxisPixel - startYAxisPixel

    startYAxisRatio, endYAxisRatio = curveObject['yAxisRatioRange']
    yAxisRatioRange = endYAxisRatio - startYAxisRatio

    maxX, maxY = np.max(curve, axis=0)

    x = ((curve[:, 0] / maxX) * wavelengthRange) + startWavelength
    y = ((curve[:, 1] / yAxisPixelRange) * yAxisRatioRange) + startYAxisRatio

    curveObject['curve'] = np.stack([x, y], axis=1)
    return curveObject

def cropCurve(curveObject):
    curve = curveObject['curve']
    start, end = targetWavelengthRange
    startIndex = np.argmax(curve[:, 0] == start)
    endIndex = np.argmax(curve[:, 0] == end)
    curve = curve[startIndex:endIndex, :]
    curveObject['curve'] = curve
    curveObject['wavelengthRange'] = targetWavelengthRange
    return curveObject

def plotCurve(curveObject, marker, show=True):
    curve = curveObject['curve']
    plt.plot(curve[:, 0], curve[:, 1], marker)

    if show:
        plt.show()

def invertCurve(curveObject):
    curve = curveObject['curve']
    startYAxisPixel, endYAxisPixel = curveObject['yAxisPixelRange']
    curve[:, 1] = endYAxisPixel - curve[:, 1]
    curveObject['curve'] = curve
    return curveObject

def getCountryCurveObject(name):
    countryObject = readTracedCurves(name)
    correctedOrientation = invertCurve(countryObject)
    scaled = scaleCurve(correctedOrientation)
    quantized = quantizeCurve(scaled)
    smoothed = smoothCurve(quantized)
    cropped = cropCurve(smoothed)
    return cropped

def getGroundtruthSunlightCurveObject():
    sunlightObject = readTracedCurves('sunlight')
    scaled = scaleCurve(sunlightObject)
    quantized = quantizeCurve(scaled)
    smoothed = smoothCurve(quantized)
    cropped = cropCurve(smoothed)
    return cropped

#europe1 = getCountryCurveObject('europe1')
#europe2 = getCountryCurveObject('europe2')
#europe3 = getCountryCurveObject('europe3')
#southAsia1 = getCountryCurveObject('southAsia1')
#southAsia2 = getCountryCurveObject('southAsia2')
#southAsia3 = getCountryCurveObject('southAsia3')
#eastAsia1 = getCountryCurveObject('eastAsia1')
#eastAsia2 = getCountryCurveObject('eastAsia2')
#eastAsia3 = getCountryCurveObject('eastAsia3')
#africa1 = getCountryCurveObject('africa1')
#africa2 = getCountryCurveObject('africa2')
#africa3 = getCountryCurveObject('africa3')
#
#groundTruthSunlight = getGroundtruthSunlightCurveObject()
#
#plotCurve(europe1, 'r-', False)
#plotCurve(europe2, 'b-', False)
#plotCurve(europe3, 'g-', False)
#plotCurve(southAsia1, 'r-', False)
#plotCurve(southAsia2, 'b-', False)
#plotCurve(southAsia3, 'g-', False)
#plotCurve(eastAsia1, 'r-', False)
#plotCurve(eastAsia2, 'b-', False)
#plotCurve(eastAsia3, 'g-', False)
#plotCurve(africa1, 'r-', False)
#plotCurve(africa2, 'b-', False)
#plotCurve(africa3, 'g-', False)
#
#plotCurve(groundTruthSunlight, 'y-')

