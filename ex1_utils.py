"""
        '########:'##::::'##::::'##:::
         ##.....::. ##::'##:::'####:::
         ##::::::::. ##'##::::.. ##:::
         ######:::::. ###::::::: ##:::
         ##...:::::: ## ##:::::: ##:::
         ##:::::::: ##:. ##::::: ##:::
         ########: ##:::. ##::'######:
        ........::..:::::..:::......::
"""
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import cv2

LOAD_GRAY_SCALE = 1
LOAD_RGB = 2


def myID() -> int:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """
    return 200000000


def imReadAndConvert(filename: str, representation: int) -> np.ndarray:
    """
    Reads an image, and returns the image converted as requested
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: The image object
    """

    # Reading the img file
    originalImg = cv2.imread(filename)

    # Convert the given image to the wished format
    if representation is LOAD_GRAY_SCALE:
        newImg = cv2.cvtColor(originalImg, cv2.COLOR_BGR2GRAY)
    elif representation is LOAD_RGB:
        newImg = cv2.cvtColor(originalImg, cv2.COLOR_BGR2RGB)
    # else error

    return (newImg / float(255)).copy()


def imDisplay(filename: str, representation: int):
    """
    Reads an image as RGB or GRAY_SCALE and displays it
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: None
    """

    # Reading the img file
    img = imReadAndConvert(filename, representation)

    # Show the wished image on the screen
    plt.imshow(img)

    # If the given representation is gray, make the img gray
    if representation is LOAD_GRAY_SCALE:
        plt.gray()
    plt.show()


def transformRGB2YIQ(imgRGB: np.ndarray) -> np.ndarray:
    """
    Converts an RGB image to YIQ color space
    :param imgRGB: An Image in RGB
    :return: A YIQ in image color space
    """

    # Using the well-known matrix to convert RGB to YIQ
    convert_matrix = np.array([[0.299, 0.587, 0.114], [0.596, -0.274, -0.322], [0.211, -0.523, 0.312]])

    # Multiplying the matrices to grant the wished new matrix
    imgYIQ = imgRGB.dot(convert_matrix.T.copy())

    # Check bounds. If some pixels are out of bounds, set them to the limits.
    imgYIQ[imgYIQ < 0] = 0
    imgYIQ[imgYIQ > 1] = 1

    return imgYIQ.copy()


def transformYIQ2RGB(imgYIQ: np.ndarray) -> np.ndarray:
    """
    Converts an YIQ image to RGB color space
    :param imgYIQ: An Image in YIQ
    :return: A RGB in image color space
    """

    # Using the well-known matrix to convert YIQ to RGB (Inverse of the original)
    convert_matrix = np.array([[0.299, 0.587, 0.114], [0.596, -0.274, -0.322], [0.211, -0.523, 0.312]])
    convert_matrix = np.linalg.inv(convert_matrix)

    # Multiplying the matrices to grant the wished new matrix
    imgRGB = imgYIQ.dot(convert_matrix.T)

    # Check bounds. If some pixels are out of bounds, set them to the limits.
    imgRGB[imgRGB < 0] = 0
    imgRGB[imgRGB > 1] = 1

    return imgRGB.copy()


def hsitogramEqualize(imgOrig: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    """
        Equalizes the histogram of an image
        :param imgOrig: Original Histogram
        :return: (imgEq,histOrg,histEQ)
    """
    
    pass


def quantizeImage(imOrig: np.ndarray, nQuant: int, nIter: int) -> (List[np.ndarray], List[float]):
    """
        Quantized an image in to **nQuant** colors
        :param imOrig: The original image (RGB or Gray scale)
        :param nQuant: Number of colors to quantize the image to
        :param nIter: Number of optimization loops
        :return: (List[qImage_i],List[error_i])
    """
    pass
