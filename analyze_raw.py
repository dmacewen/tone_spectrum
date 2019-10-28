import cv2
import rawpy
import numpy as np


#imgName = 'sky5'
#imgName = 'sunRef2'
#imgName = 'sunRef3'
#imgName = 'sunRef4'
#imgName = 'sunRef5'
#imgName = 'sunRef7'
#imgName = 'iPad1'
imgName = 'iPad_ScreenFlash1'
#imgName = 'benQ1'


def stretch(img):
    minVal = np.min(img)
    maxVal = np.max(img)

    return (img - minVal) / (maxVal - minVal)

with rawpy.imread('images/{}.DNG'.format(imgName)) as raw:
    print('Pattern :: \n{}'.format(raw.raw_pattern))
    print('Num Colors :: \n{}'.format(raw.num_colors))
    print('Color Mat :: \n{}'.format(raw.color_matrix))
    print('Color Desc :: \n{}'.format(raw.color_desc))
    print('Img Colors :: \n{}'.format(raw.raw_colors))
    print('Img Colors Shape :: \n{}'.format(raw.raw_colors.shape))
    print('------')
    
    redMask = raw.raw_colors == 0
    #greenMask = np.logical_or((raw.raw_colors == 1), (raw.raw_colors == 3))
    greenMask = raw.raw_colors == 1
    blueMask = raw.raw_colors == 2

    h, w = raw.raw_image.shape

    #y = 0.48
    #y = 0.53
    y = 0.56
    numbersY = y - .035
    x = 0.6

    #targetHeight = 0.1
    targetHeight = 0.04
    targetNumbersHeight = 0.03
    targetWidth = 0.3

    numbers_start_height = int(numbersY * h)
    numbers_end_height = int((numbersY + targetNumbersHeight) * h)


    start_height = int(y * h)
    end_height = int((y + targetHeight) * h)
    start_width = int(x * w)
    end_width = int((x + targetWidth) * w)

    cropped = np.copy(raw.raw_image[start_height:end_height, start_width:end_width])

    numbersCropped = np.copy(raw.raw_image[numbers_start_height:numbers_end_height, start_width:end_width])
    numbersCropped = stretch(numbersCropped)

    redMask = redMask[start_height:end_height, start_width:end_width]
    greenMask = greenMask[start_height:end_height, start_width:end_width]
    blueMask = blueMask[start_height:end_height, start_width:end_width]

    stretched = stretch(cropped)

    redImg = np.copy(stretched)
    redImg[np.logical_not(redMask)] = 0
    #redImg = stretch(redImg)

    greenImg = np.copy(stretched)
    greenImg[np.logical_not(greenMask)] = 0
    #greenImg = stretch(greenImg)

    blueImg = np.copy(stretched)
    blueImg[np.logical_not(blueMask)] = 0
    #blueImg = stretch(blueImg)

    print('Raw Image')
    print(raw.raw_image.shape)

    #cv2.imshow('Raw', raw.raw_image)

    imgs = np.vstack([numbersCropped, redImg, greenImg, blueImg])
    #cv2.imshow('Red', redImg)
    #cv2.imshow('Green', greenImg)
    #cv2.imshow('Blue', blueImg)
    cv2.imshow('RGB', imgs)
    cv2.waitKey(0)
