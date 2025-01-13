import cv2
import matplotlib.pyplot as plt
import numpy as np


def histogram(smooth_img):
    img = smooth_img
    r, c = img.shape
    hist = np.zeros(256)
    for i in range(r):
        for j in range(c):
            inten = img[i, j]
            hist[inten] += 1
    PDF = hist / (r * c)

    CDF = np.zeros(256)
    temp = 0
    for i in range(256):
        temp += PDF[i]
        CDF[i] = temp
    CDF *= 255

    # Global Histogram Equalization
    out = np.zeros((r, c))
    for i in range(r):
        for j in range(c):
            inten = img[i, j]
            p = np.round(CDF[inten])
            out[i, j] = p

    out = cv2.normalize(out, None, 0, 1, cv2.NORM_MINMAX) * 255
    out = out.astype(np.uint8)


    # CLAHE (Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(img)


    #PDF and CDF for CLAHE image
    output_hist = np.zeros(256)
    for i in range(r):
        for j in range(c):
            inten = clahe_img[i, j]
            output_hist[inten] += 1
    PDF = output_hist / (r * c)

    CDF = np.zeros(256)
    temp = 0
    for i in range(256):
        temp += PDF[i]
        CDF[i] = temp
    CDF *= 255


    return out,clahe_img


