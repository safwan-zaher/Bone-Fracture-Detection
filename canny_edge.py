# canny_edge.py

import numpy as np
import cv2

def normalize(out):
    cv2.normalize(out, out, 0, 255, cv2.NORM_MINMAX)
    out = np.round(out).astype(np.uint8)
    return out

def non_maximum_suppression(image, angle):
    image = image / image.max() * 255
    out = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            q = 0
            r = 0
            ang = angle[i, j]
            if (-22.5 <= ang < 22.5) or (157.5 <= ang <= 180) or (-180 <= ang <= -157.5):
                r = image[i, j - 1]
                q = image[i, j + 1]
            elif (-67.5 <= ang <= -22.5) or (112.5 <= ang <= 157.5):
                r = image[i - 1, j + 1]
                q = image[i + 1, j - 1]
            elif (67.5 <= ang <= 112.5) or (-112.5 <= ang <= -67.5):
                r = image[i - 1, j]
                q = image[i + 1, j]
            elif (22.5 <= ang < 67.5) or (-167.5 <= ang <= -112.5):
                r = image[i + 1, j + 1]
                q = image[i - 1, j - 1]

            if (image[i, j] >= q) and (image[i, j] >= r):
                out[i, j] = image[i, j]
            else:
                out[i, j] = 0

    return out

def threshold(img, lowThresholdRatio=0.05, highThresholdRatio=0.09):
    highThreshold = img.max() * highThresholdRatio
    lowThreshold = highThreshold * lowThresholdRatio

    M, N = img.shape
    res = np.zeros((M, N), dtype=np.int32)

    weak = np.int32(25)
    strong = np.int32(255)

    strong_i, strong_j = np.where(img >= highThreshold)
    zeros_i, zeros_j = np.where(img < lowThreshold)

    weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold))

    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak

    return res, weak, strong

def hysteresis(image, weak, strong=255):
    M, N = image.shape
    out = image.copy()

    for i in range(1, M - 1):
        for j in range(1, N - 1):
            if (image[i, j] == weak):
                if np.any(image[i - 1:i + 2, j - 1:j + 2] == strong):
                    out[i, j] = strong
                else:
                    out[i, j] = 0
    return out
