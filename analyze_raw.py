import cv2
import rawpy
import numpy as np
import matplotlib.pyplot as plt


#imgName = 'sky5'
#imgName = 'sunRef2'
#imgName = 'sunRef3'
#imgName = 'sunRef4'
#imgName = 'sunRef5'
#imgName = 'sunRef7'
#imgName = 'iPad1'
#imgName = 'iPad_ScreenFlash1'
imgName = 'benQ1'


def stretch(img, mask=None):
    mask = mask if mask is not None else np.ones(img.shape, dtype='bool')
    minVal = np.min(img[mask])
    maxVal = np.max(img[mask])

    spread = maxVal - minVal
    stretched = (img - minVal) / spread
    stretched[np.logical_not(mask)] = 0
    return stretched, spread

def autocrop(img, redMask, greenMask, blueMask):
    scaled, _ = stretch(img)

    threshold = 0.95
    brightSubpixelMask = scaled > threshold
    brightSubpixelMask = brightSubpixelMask.astype('uint8') * 255

    morphologyKernel = np.ones((21, 21), np.uint8)
    dilated = cv2.dilate(brightSubpixelMask, morphologyKernel, iterations=1)

    contours, heirarchy = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(contour) for contour in contours]
    largestContour = np.argmax(areas)

    numbersBB = cv2.boundingRect(contours[largestContour])
    #spectrumBB = numbersBB + np.array([0, int(1.5 * numbersBB[3]), 0, 0)]) #Just offset the numberline by 2 times its height. Samples roughly the middle of the spectrum area
    spectrumBB = numbersBB + np.array([0, int(2 * numbersBB[3]), 0, int(-0.5 * numbersBB[3])]) #Just offset the numberline by 2 times its height. Samples roughly the middle of the spectrum area

    return [numbersBB, spectrumBB]

with rawpy.imread('images/{}.DNG'.format(imgName)) as raw:
    redMask = raw.raw_colors == 0
    #greenMask = np.logical_or((raw.raw_colors == 1), (raw.raw_colors == 3))
    greenMask = raw.raw_colors == 1
    blueMask = raw.raw_colors == 2

    numbersBB, spectrumBB = autocrop(raw.raw_image, redMask, greenMask, blueMask)

    numbers = np.copy(raw.raw_image[numbersBB[1]:(numbersBB[1] + numbersBB[3]), numbersBB[0]:(numbersBB[0] + numbersBB[2])])
    spectrum = np.copy(raw.raw_image[spectrumBB[1]:(spectrumBB[1] + spectrumBB[3]), spectrumBB[0]:(spectrumBB[0] + spectrumBB[2])])

    redMask = redMask[spectrumBB[1]:(spectrumBB[1] + spectrumBB[3]), spectrumBB[0]:(spectrumBB[0] + spectrumBB[2])]
    greenMask = greenMask[spectrumBB[1]:(spectrumBB[1] + spectrumBB[3]), spectrumBB[0]:(spectrumBB[0] + spectrumBB[2])]
    blueMask = blueMask[spectrumBB[1]:(spectrumBB[1] + spectrumBB[3]), spectrumBB[0]:(spectrumBB[0] + spectrumBB[2])]

    numbers, _ = stretch(numbers)

    redImg = np.copy(spectrum)
    redImg, redRange = stretch(redImg, redMask)
    print('Red Range :: {}'.format(redRange))
    redRowMask = np.any(redImg.T, axis=0)
    redMedians = np.median(redImg.T[:, redRowMask], axis=1)
    redMedians = redMedians[redMedians > 0]

    greenImg = np.copy(spectrum)
    greenImg, greenRange = stretch(greenImg, greenMask)
    print('Green Range :: {}'.format(greenRange))
    greenRowMask = np.any(greenImg.T, axis=0)
    greenMedians = np.median(greenImg.T[:, greenRowMask], axis=1)
    greenMedians = greenMedians[greenMedians > 0]

    blueImg = np.copy(spectrum)
    blueImg, blueRange = stretch(blueImg, blueMask)
    print('Blue Range :: {}'.format(blueRange))
    blueRowMask = np.any(blueImg.T, axis=0)
    blueMedians = np.median(blueImg.T[:, blueRowMask], axis=1)
    blueMedians = blueMedians[blueMedians > 0]


    fig = plt.figure()
    ax1 = fig.add_axes([0.1, 0.5, 0.85, 0.5])
    ax1.plot(np.arange(len(redMedians)), redMedians, 'r-')
    ax1.plot(np.arange(len(greenMedians)), greenMedians, 'g-')
    ax1.plot(np.arange(len(blueMedians)), blueMedians, 'b-')

    ax2 = fig.add_axes([0.1, 0, 0.85, 0.5])
    ax2.imshow(np.vstack([numbers, redImg, greenImg, blueImg]), cmap='gray')#, interpolation='nearest')

    fig.show()

    imgs = np.vstack([numbers, redImg, greenImg, blueImg])
    cv2.imshow('RGB', imgs)
    cv2.waitKey(0)
