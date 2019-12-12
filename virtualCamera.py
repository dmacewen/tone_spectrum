import numpy as np
import spectrumTools
import copy
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
    print('ground truth sunlight curve :: {}'.format(groundTruthCorrectionArray))

    redMeasuredSunlightCurveObject, greenMeasuredSunlightCurveObject, blueMeasuredSunlightCurveObject = measuredSunlightRGBCurvesObjects

    redSensitivityY = redMeasuredSunlightCurveObject['curve'][:, 1] * groundTruthCorrectionArray
    greenSensitivityY = greenMeasuredSunlightCurveObject['curve'][:, 1] * groundTruthCorrectionArray
    blueSensitivityY = blueMeasuredSunlightCurveObject['curve'][:, 1] * groundTruthCorrectionArray

    redSensitivity = np.stack([redMeasuredSunlightCurveObject['curve'][:, 0], redSensitivityY], axis=1)
    greenSensitivity = np.stack([greenMeasuredSunlightCurveObject['curve'][:, 0], greenSensitivityY], axis=1)
    blueSensitivity = np.stack([blueMeasuredSunlightCurveObject['curve'][:, 0], blueSensitivityY], axis=1)



    redSensitivityCurveObject = spectrumTools.makeCurveObject(redSensitivity, redMeasuredSunlightCurveObject['wavelengthRange'], redMeasuredSunlightCurveObject['yAxisPixelRange'], redMeasuredSunlightCurveObject['yAxisRatioRange'])
    greenSensitivityCurveObject = spectrumTools.makeCurveObject(greenSensitivity, greenMeasuredSunlightCurveObject['wavelengthRange'], greenMeasuredSunlightCurveObject['yAxisPixelRange'], greenMeasuredSunlightCurveObject['yAxisRatioRange'])
    blueSensitivityCurveObject = spectrumTools.makeCurveObject(blueSensitivity, blueMeasuredSunlightCurveObject['wavelengthRange'], blueMeasuredSunlightCurveObject['yAxisPixelRange'], blueMeasuredSunlightCurveObject['yAxisRatioRange'])

    return [redSensitivityCurveObject, greenSensitivityCurveObject, blueSensitivityCurveObject]


def illuminateSurface(lightSourceCurveObject, surfaceCurveObject):
    lightSourceCurve = lightSourceCurveObject['curve']
    surfaceCurve = surfaceCurveObject['curve']

    diffuseReflectedCurve = np.stack([lightSourceCurve[:, 0], (lightSourceCurve[:, 1] * surfaceCurve[:, 1])], axis=1)

    diffuseReflectionCurveObject = np.copy(lightSourceCurveObject)
    diffuseReflectionCurveObject['curve'] = diffuseReflectedCurve

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

    #plt.plot(spectrum[:, 0], red, 'r-')
    #plt.plot(spectrum[:, 0], green, 'g-')
    #plt.plot(spectrum[:, 0], blue, 'b-')
    #plt.show()

    sums = [np.sum(red), np.sum(green), np.sum(blue)]
    largest = max(sums)
    scaledSums = sums / largest

    scaledSums8bit = [np.round(channel * 255).astype('uint8') for channel in scaledSums]

    #print('RGB :: ({}, {}, {})'.format(blue, green, red))
    #print('----------')
    #print('RGB :: ({}, {}, {})'.format(*sums))
    #print('Scaled RGB :: ({}, {}, {})'.format(*scaledSums))
    #print('Scaled RGB 8bit :: ({}, {}, {})'.format(*scaledSums8bit))
    #print('----------')
    return [blue, green, red]

SensorSensitivities['iphoneX'] = getSensorSensitivity(spectrumTools.rgbSunCurves, spectrumTools.groundTruthSunlight)


ledSpectrum = spectrumTools.getLightSourceCurve(LightSources['LED'])

recordRGBValues(ledSpectrum, scaledEyeSensitivity['iphoneX'])
