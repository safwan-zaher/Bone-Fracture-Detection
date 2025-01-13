import math
import numpy as np
import cv2

def median_filter(img, kernel_size):
    m = kernel_size//2
    img_bordered = cv2.copyMakeBorder(img, m, m, m, m, borderType=cv2.BORDER_CONSTANT, value=0)
    out = np.zeros_like(img)
    height, width = img.shape
    for i in range(m, height + m):
        for j in range(m, width + m):
            region = img_bordered[i-m:i+m+1, j-m:j+m+1]
            out[i-m, j-m] = np.median(region)

    return out

def gaussian_kernel(height,width,sigmaX,sigmaY):
    kernel = np.zeros((height, width), dtype=np.float32)
    x = height // 2
    y = width // 2
    const = 1 / (2 * math.pi * sigmaX * sigmaY)
    sigmaX2 = sigmaX ** 2
    sigmaY2 = sigmaY ** 2

    for p in range(-x, x + 1):
        for q in range(-y, y + 1):
            kernel[p + x, q + y] = const * math.exp(-0.5 * (p ** 2 / sigmaX2 + q ** 2 / sigmaY2))

    return kernel / np.sum(kernel)

def convolution(title, kernel, img):
    m = kernel.shape[0] // 2
    n = kernel.shape[1] // 2

    img_bordered = cv2.copyMakeBorder(img, m, m, n, n, borderType=cv2.BORDER_CONSTANT, value=0)

    out = np.zeros_like(img)
    height, width = img.shape

    for i in range(m, height + m):
        for j in range(n, width + n):
            region = img_bordered[i-m:i+m+1, j-n:j+n+1]
            out[i-m, j-n] = np.sum(region * kernel)

    out = np.clip(out, 0, 255)
    out = out.astype(np.uint8)

    return out


