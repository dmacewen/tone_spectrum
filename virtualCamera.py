import numpy as np
import spectrumTools
import colorSpaceTools
import copy
from pprint import pprint
import matplotlib.pyplot as plt

#Measured
LightSources = {}
LightSources['LED'] = 'led'
LightSources['IncandecentA'] = 'incA'
LightSources['IncandecentB'] = 'incB'
LightSources['BenQ'] = 'BenQ'
LightSources['Sky'] = 'Sky'
LightSources['iPad'] = 'iPad'
LightSources['Sun'] = 'Sun'

#Traced
Surfaces = {}
Surfaces['europe'] = ['europe1','europe2','europe3']
Surfaces['southAsia'] = ['southAsia1','southAsia2','southAsia3']
Surfaces['eastAsia'] = ['eastAsia1','eastAsia2','eastAsia3']
Surfaces['africa'] = ['africa1','africa2','africa3']

SensorSensitivities = {}

def getSensorSensitivity(measuredSunlightRGBCurvesObjects, groundTruthSunlightCurveObject):
    groundTruthCorrectionArray = (1 / groundTruthSunlightCurveObject['curve'][:, 1])

    redMeasuredSunlightCurveObject, greenMeasuredSunlightCurveObject, blueMeasuredSunlightCurveObject = measuredSunlightRGBCurvesObjects

    redSensitivityY = redMeasuredSunlightCurveObject['curve'][:, 1] * groundTruthCorrectionArray
    greenSensitivityY = greenMeasuredSunlightCurveObject['curve'][:, 1] * groundTruthCorrectionArray
    blueSensitivityY = blueMeasuredSunlightCurveObject['curve'][:, 1] * groundTruthCorrectionArray

    redSensitivity = np.stack([redMeasuredSunlightCurveObject['curve'][:, 0], redSensitivityY], axis=1)
    greenSensitivity = np.stack([greenMeasuredSunlightCurveObject['curve'][:, 0], greenSensitivityY], axis=1)
    blueSensitivity = np.stack([blueMeasuredSunlightCurveObject['curve'][:, 0], blueSensitivityY], axis=1)


    redSensitivityCurveObject = spectrumTools.copyCurveObject(redSensitivity, redMeasuredSunlightCurveObject)
    greenSensitivityCurveObject = spectrumTools.copyCurveObject(greenSensitivity, greenMeasuredSunlightCurveObject)
    blueSensitivityCurveObject = spectrumTools.copyCurveObject(blueSensitivity, blueMeasuredSunlightCurveObject)

    return [redSensitivityCurveObject, greenSensitivityCurveObject, blueSensitivityCurveObject]


def illuminateSurface(lightSourceCurveObject, surfaceCurveObject):
    lightSourceCurve = lightSourceCurveObject['curve']
    surfaceCurve = surfaceCurveObject['curve']

    diffuseReflectedCurve = np.stack([lightSourceCurve[:, 0], (lightSourceCurve[:, 1] * surfaceCurve[:, 1])], axis=1)
    diffuseReflectionCurveObject = spectrumTools.copyCurveObject(diffuseReflectedCurve, surfaceCurveObject)

    return diffuseReflectionCurveObject

def recordRGBValues(spectrumObject, rgbSensitivityCurveObjects):
    spectrumCurve = spectrumObject['curve']

    redSensitivityCurveObject, greenSensitivityCurveObject, blueSensitivityCurveObject = rgbSensitivityCurveObjects

    redSensitivityCurve = redSensitivityCurveObject['curve']
    greenSensitivityCurve = greenSensitivityCurveObject['curve']
    blueSensitivityCurve = blueSensitivityCurveObject['curve']

    red = spectrumCurve[:, 1] * redSensitivityCurve[:, 1]
    green = spectrumCurve[:, 1] * greenSensitivityCurve[:, 1]
    blue = spectrumCurve[:, 1] * blueSensitivityCurve[:, 1]

    sums = [np.sum(red), np.sum(green), np.sum(blue)]
    largest = max(sums)
    scaledSums = sums / largest

    scaledSums8bit = [np.round(channel * 255).astype('uint8') for channel in scaledSums]

    return scaledSums8bit

def whiteBalance(rgbValues, whiteBalanceMultiplier):
    balanced = rgbValues * whiteBalanceMultiplier
    return balanced / max(balanced)

def cleanRGBTriplet(rgb):
    return list(np.floor(rgb * 255).astype('uint8'))

def cleanLABTriplet(lab):
    return list(lab)

SensorSensitivities['iphoneX'] = getSensorSensitivity(spectrumTools.rgbSunCurves, spectrumTools.groundTruthSunlight)

ledSpectrum = spectrumTools.getLightSourceCurve(LightSources['LED'])
incASpectrum = spectrumTools.getLightSourceCurve(LightSources['IncandecentA'])
sunSpectrum = spectrumTools.getLightSourceCurve(LightSources['Sun'])
iPadSpectrum = spectrumTools.getLightSourceCurve(LightSources['iPad'])

whitePoint = recordRGBValues(sunSpectrum, SensorSensitivities['iphoneX'])
whiteBalanceMultiplier = 1 / (whitePoint / max(whitePoint))

def exposeSurfaceToLight(surface, sensor, incedentLight):
    reflection = illuminateSurface(incedentLight, surface)
    rgbTriplet = recordRGBValues(reflection, sensor)
    wbRGBTriplet = whiteBalance(rgbTriplet, whiteBalanceMultiplier)
    lab = colorSpaceTools.rgb_to_lab(wbRGBTriplet)
    #print('{} -> {}'.format(list(wbRGBTriplet), list(lab)))
    return cleanLABTriplet(lab)


def exposeSurfaceToAllLights(surface, sensor):
    results = {}
    results['led'] = exposeSurfaceToLight(ledSpectrum, sensor, surface)
    results['incA'] = exposeSurfaceToLight(incASpectrum, sensor, surface)
    results['sun'] = exposeSurfaceToLight(sunSpectrum, sensor, surface)
    results['iPad'] = exposeSurfaceToLight(iPadSpectrum, sensor, surface) 
    return results

def calcualteDistance(lab1, lab2):
    labDiff = np.array(lab2) - np.array(lab1)
    return np.sqrt(np.sum(labDiff * labDiff))

def compareLightSources(surface1, surface2, lightSource1, lightSource2):
    s1_s2_l1 = calcualteDistance(surface1[lightSource1], surface2[lightSource1])
    s1_s2_l2 = calcualteDistance(surface1[lightSource2], surface2[lightSource2])
    return [s1_s2_l1, s1_s2_l2]

def regionalComparison(surface1, surface2, surface3, lightSource1, lightSource2):
    s1s2 = compareLightSources(surface1, surface2, lightSource1, lightSource2)
    s2s3 = compareLightSources(surface2, surface3, lightSource1, lightSource2)
    s1s3 = compareLightSources(surface1, surface3, lightSource1, lightSource2)

    print('S1 vs S2 | iPad vs Sun :: {} vs {}'.format(*s1s2))
    print('S2 vs S3 | iPad vs Sun :: {} vs {}'.format(*s2s3))
    print('S1 vs S3 | iPad vs Sun :: {} vs {}'.format(*s1s3))

print('----- Europe 1 ----')
europe1 = spectrumTools.getCountryCurveObject(Surfaces['europe'][0])
europe1Results = exposeSurfaceToAllLights(europe1, SensorSensitivities['iphoneX'])
pprint(europe1Results)

print('----- Europe 2 ----')
europe2 = spectrumTools.getCountryCurveObject(Surfaces['europe'][1])
europe2Results = exposeSurfaceToAllLights(europe2, SensorSensitivities['iphoneX'])
pprint(europe2Results)

print('----- Europe 3 ----')
europe3 = spectrumTools.getCountryCurveObject(Surfaces['europe'][2])
europe3Results = exposeSurfaceToAllLights(europe3, SensorSensitivities['iphoneX'])
pprint(europe3Results)

print('----- SouthAsia 1 ----')
southAsia1 = spectrumTools.getCountryCurveObject(Surfaces['southAsia'][0])
southAsia1Results = exposeSurfaceToAllLights(southAsia1, SensorSensitivities['iphoneX'])
pprint(southAsia1Results)

print('----- SouthAsia 2 ----')
southAsia2 = spectrumTools.getCountryCurveObject(Surfaces['southAsia'][1])
southAsia2Results = exposeSurfaceToAllLights(southAsia2, SensorSensitivities['iphoneX'])
pprint(southAsia2Results)

print('----- SouthAsia 3 ----')
southAsia3 = spectrumTools.getCountryCurveObject(Surfaces['southAsia'][2])
southAsia3Results = exposeSurfaceToAllLights(southAsia3, SensorSensitivities['iphoneX'])
pprint(southAsia3Results)

print('----- EastAsia 1 ----')
eastAsia1 = spectrumTools.getCountryCurveObject(Surfaces['eastAsia'][0])
eastAsia1Results = exposeSurfaceToAllLights(eastAsia1, SensorSensitivities['iphoneX'])
pprint(eastAsia1Results)

print('----- EastAsia 2 ----')
eastAsia2 = spectrumTools.getCountryCurveObject(Surfaces['eastAsia'][1])
eastAsia2Results = exposeSurfaceToAllLights(eastAsia2, SensorSensitivities['iphoneX'])
pprint(eastAsia2Results)

print('----- EastAsia 3 ----')
eastAsia3 = spectrumTools.getCountryCurveObject(Surfaces['eastAsia'][2])
eastAsia3Results = exposeSurfaceToAllLights(eastAsia3, SensorSensitivities['iphoneX'])
pprint(eastAsia3Results)

print('----- Africa 1 ----')
africa1 = spectrumTools.getCountryCurveObject(Surfaces['africa'][0])
africa1Results = exposeSurfaceToAllLights(africa1, SensorSensitivities['iphoneX'])
pprint(africa1Results)

print('----- Africa 2 ----')
africa2 = spectrumTools.getCountryCurveObject(Surfaces['africa'][1])
africa2Results = exposeSurfaceToAllLights(africa2, SensorSensitivities['iphoneX'])
pprint(africa2Results)

print('----- Africa 3 ----')
africa3 = spectrumTools.getCountryCurveObject(Surfaces['africa'][2])
africa3Results = exposeSurfaceToAllLights(africa3, SensorSensitivities['iphoneX'])
pprint(africa3Results)

print('------\n------')

print('EUROPE')
regionalComparison(europe1Results, europe2Results, europe3Results, 'iPad', 'sun')
print('EAST ASIA')
regionalComparison(eastAsia1Results, eastAsia2Results, eastAsia3Results, 'iPad', 'sun')
print('SOUTH ASIA')
regionalComparison(southAsia1Results, southAsia2Results, southAsia3Results, 'iPad', 'sun')
print('AFRICA')
regionalComparison(africa1Results, africa2Results, africa3Results, 'iPad', 'sun')

