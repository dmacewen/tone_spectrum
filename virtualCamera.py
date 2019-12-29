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
SensorSensitivities['iphoneX'] = {}
SensorSensitivities['humanEye'] = {}
SensorSensitivities['humanEye']['curves'] = spectrumTools.getEyeCurveObjects()

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

def scaleToMax(rgbValues):
    return rgbValues / max(rgbValues)

def cleanRGBTriplet(rgb):
    return list(np.floor(rgb * 255).astype('uint8'))

def cleanLABTriplet(lab):
    return list(lab)

SensorSensitivities['iphoneX']['curves'] = getSensorSensitivity(spectrumTools.rgbSunCurves, spectrumTools.groundTruthSunlight)

ledSpectrum = spectrumTools.getLightSourceCurve(LightSources['LED'])
incASpectrum = spectrumTools.getLightSourceCurve(LightSources['IncandecentA'])
sunSpectrum = spectrumTools.getLightSourceCurve(LightSources['Sun'])
iPadSpectrum = spectrumTools.getLightSourceCurve(LightSources['iPad'])

SensorSensitivities['iphoneX']['whitePoint'] = recordRGBValues(sunSpectrum, SensorSensitivities['iphoneX']['curves'])
SensorSensitivities['iphoneX']['whiteBalanceMultiplier'] = 1 / (SensorSensitivities['iphoneX']['whitePoint'] / max(SensorSensitivities['iphoneX']['whitePoint']))

SensorSensitivities['humanEye']['whitePoint'] = recordRGBValues(sunSpectrum, SensorSensitivities['humanEye']['curves'])
SensorSensitivities['humanEye']['whiteBalanceMultiplier'] = 1 / (SensorSensitivities['humanEye']['whitePoint'] / max(SensorSensitivities['humanEye']['whitePoint']))

def exposeSurfaceToLight(surface, sensor, incedentLight):
    reflection = illuminateSurface(incedentLight, surface)
    rgbTriplet = recordRGBValues(reflection, sensor['curves'])
    print('RGB Triplet :: {}'.format(rgbTriplet))
    #wbRGBTriplet = whiteBalance(rgbTriplet, sensor['whiteBalanceMultiplier'])
    #scaledRGBTriplet = scaleToMax(rgbTriplet)

    #lab = colorSpaceTools.rgb_to_lab(wbRGBTriplet)
    lab = colorSpaceTools.rgb_to_lab(rgbTriplet)
    print('LAB Triplet :: {}'.format(rgbTriplet))
    return cleanLABTriplet(lab)


def exposeSurfaceToAllLights(surface, sensor):
    results = {}
    results['led'] = exposeSurfaceToLight(ledSpectrum, sensor, surface)
    results['incA'] = exposeSurfaceToLight(incASpectrum, sensor, surface)
    results['sun'] = exposeSurfaceToLight(sunSpectrum, sensor, surface)
    results['iPad'] = exposeSurfaceToLight(iPadSpectrum, sensor, surface) 
    return results

def calculateDistance(lab1, lab2):
    labDiff = np.array(lab2) - np.array(lab1)
    return np.sqrt(np.sum(labDiff * labDiff))

def compareLightSources(surface1, surface2, lightSource1, lightSource2):
    s1_s2_l1 = calculateDistance(surface1[lightSource1], surface2[lightSource1])
    s1_s2_l2 = calculateDistance(surface1[lightSource2], surface2[lightSource2])
    return [s1_s2_l1, s1_s2_l2]

def regionalComparison(surface1, surface2, surface3, lightSource1, lightSource2):
    s1s2 = compareLightSources(surface1, surface2, lightSource1, lightSource2)
    s2s3 = compareLightSources(surface2, surface3, lightSource1, lightSource2)
    s1s3 = compareLightSources(surface1, surface3, lightSource1, lightSource2)

    print('S1 vs S2 | iPad vs Sun :: {} vs {} | {}'.format(*s1s2, s1s2[0] - s1s2[1]))
    print('S2 vs S3 | iPad vs Sun :: {} vs {} | {}'.format(*s2s3, s2s3[0] - s2s3[1]))
    print('S1 vs S3 | iPad vs Sun :: {} vs {} | {}'.format(*s1s3, s1s3[0] - s1s3[1]))

print('----- Europe 1 ----')
europe1 = spectrumTools.getCountryCurveObject(Surfaces['europe'][0])
europe1Results_iphoneX = exposeSurfaceToAllLights(europe1, SensorSensitivities['iphoneX'])
europe1Results_humanEye = exposeSurfaceToAllLights(europe1, SensorSensitivities['humanEye'])
pprint(europe1Results_iphoneX)
pprint(europe1Results_humanEye)

print('----- Europe 2 ----')
europe2 = spectrumTools.getCountryCurveObject(Surfaces['europe'][1])
europe2Results_iphoneX = exposeSurfaceToAllLights(europe2, SensorSensitivities['iphoneX'])
europe2Results_humanEye = exposeSurfaceToAllLights(europe2, SensorSensitivities['humanEye'])
pprint(europe2Results_iphoneX)
pprint(europe2Results_humanEye)

print('----- Europe 3 ----')
europe3 = spectrumTools.getCountryCurveObject(Surfaces['europe'][2])
europe3Results_iphoneX = exposeSurfaceToAllLights(europe3, SensorSensitivities['iphoneX'])
europe3Results_humanEye = exposeSurfaceToAllLights(europe3, SensorSensitivities['humanEye'])
pprint(europe3Results_iphoneX)
pprint(europe3Results_humanEye)

print('----- SouthAsia 1 ----')
southAsia1 = spectrumTools.getCountryCurveObject(Surfaces['southAsia'][0])
southAsia1Results_iphoneX = exposeSurfaceToAllLights(southAsia1, SensorSensitivities['iphoneX'])
southAsia1Results_humanEye = exposeSurfaceToAllLights(southAsia1, SensorSensitivities['humanEye'])
pprint(southAsia1Results_iphoneX)
pprint(southAsia1Results_humanEye)

print('----- SouthAsia 2 ----')
southAsia2 = spectrumTools.getCountryCurveObject(Surfaces['southAsia'][1])
southAsia2Results_iphoneX = exposeSurfaceToAllLights(southAsia2, SensorSensitivities['iphoneX'])
southAsia2Results_humanEye = exposeSurfaceToAllLights(southAsia2, SensorSensitivities['humanEye'])
pprint(southAsia2Results_iphoneX)
pprint(southAsia2Results_humanEye)

print('----- SouthAsia 3 ----')
southAsia3 = spectrumTools.getCountryCurveObject(Surfaces['southAsia'][2])
southAsia3Results_iphoneX = exposeSurfaceToAllLights(southAsia3, SensorSensitivities['iphoneX'])
southAsia3Results_humanEye = exposeSurfaceToAllLights(southAsia3, SensorSensitivities['humanEye'])
pprint(southAsia3Results_iphoneX)
pprint(southAsia3Results_humanEye)

print('----- EastAsia 1 ----')
eastAsia1 = spectrumTools.getCountryCurveObject(Surfaces['eastAsia'][0])
eastAsia1Results_iphoneX = exposeSurfaceToAllLights(eastAsia1, SensorSensitivities['iphoneX'])
eastAsia1Results_humanEye = exposeSurfaceToAllLights(eastAsia1, SensorSensitivities['humanEye'])
pprint(eastAsia1Results_iphoneX)
pprint(eastAsia1Results_humanEye)

print('----- EastAsia 2 ----')
eastAsia2 = spectrumTools.getCountryCurveObject(Surfaces['eastAsia'][1])
eastAsia2Results_iphoneX = exposeSurfaceToAllLights(eastAsia2, SensorSensitivities['iphoneX'])
eastAsia2Results_humanEye = exposeSurfaceToAllLights(eastAsia2, SensorSensitivities['humanEye'])
pprint(eastAsia2Results_iphoneX)
pprint(eastAsia2Results_humanEye)

print('----- EastAsia 3 ----')
eastAsia3 = spectrumTools.getCountryCurveObject(Surfaces['eastAsia'][2])
eastAsia3Results_iphoneX = exposeSurfaceToAllLights(eastAsia3, SensorSensitivities['iphoneX'])
eastAsia3Results_humanEye = exposeSurfaceToAllLights(eastAsia3, SensorSensitivities['humanEye'])
pprint(eastAsia3Results_iphoneX)
pprint(eastAsia3Results_humanEye)

print('----- Africa 1 ----')
africa1 = spectrumTools.getCountryCurveObject(Surfaces['africa'][0])
africa1Results_iphoneX = exposeSurfaceToAllLights(africa1, SensorSensitivities['iphoneX'])
africa1Results_humanEye = exposeSurfaceToAllLights(africa1, SensorSensitivities['humanEye'])
pprint(africa1Results_iphoneX)
pprint(africa1Results_humanEye)

print('----- Africa 2 ----')
africa2 = spectrumTools.getCountryCurveObject(Surfaces['africa'][1])
africa2Results_iphoneX = exposeSurfaceToAllLights(africa2, SensorSensitivities['iphoneX'])
africa2Results_humanEye = exposeSurfaceToAllLights(africa2, SensorSensitivities['humanEye'])
pprint(africa2Results_iphoneX)
pprint(africa2Results_humanEye)

print('----- Africa 3 ----')
africa3 = spectrumTools.getCountryCurveObject(Surfaces['africa'][2])
africa3Results_iphoneX = exposeSurfaceToAllLights(africa3, SensorSensitivities['iphoneX'])
africa3Results_humanEye = exposeSurfaceToAllLights(africa3, SensorSensitivities['humanEye'])
pprint(africa3Results_iphoneX)
pprint(africa3Results_humanEye)

print('------\n------')

print('EUROPE')
print('iPhoneX')
regionalComparison(europe1Results_iphoneX, europe2Results_iphoneX, europe3Results_iphoneX, 'iPad', 'sun')
print('Human Eye')
regionalComparison(europe1Results_humanEye, europe2Results_humanEye, europe3Results_humanEye, 'iPad', 'sun')
print('EAST ASIA')
print('iPhoneX')
regionalComparison(eastAsia1Results_iphoneX, eastAsia2Results_iphoneX, eastAsia3Results_iphoneX, 'iPad', 'sun')
print('Human Eye')
regionalComparison(eastAsia1Results_humanEye, eastAsia2Results_humanEye, eastAsia3Results_humanEye, 'iPad', 'sun')
print('SOUTH ASIA')
print('iPhoneX')
regionalComparison(southAsia1Results_iphoneX, southAsia2Results_iphoneX, southAsia3Results_iphoneX, 'iPad', 'sun')
print('Human Eye')
regionalComparison(southAsia1Results_humanEye, southAsia2Results_humanEye, southAsia3Results_humanEye, 'iPad', 'sun')
print('AFRICA')
print('iPhoneX')
regionalComparison(africa1Results_iphoneX, africa2Results_iphoneX, africa3Results_iphoneX, 'iPad', 'sun')
print('Human Eye')
regionalComparison(africa1Results_humanEye, africa2Results_humanEye, africa3Results_humanEye, 'iPad', 'sun')

