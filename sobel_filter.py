import numpy as np
import cv2
import matplotlib.pyplot as plt
import smoothing_filters as sf
import histogram as hist


def apply_sobel_filters(img):
    sobel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    Gradient_X = cv2.filter2D(img, -1, sobel_x)
    Gradient_Y = cv2.filter2D(img, -1, sobel_y)

    return Gradient_X, Gradient_Y

def compute_magnitude_and_direction(Gradient_X, Gradient_Y):
    G = np.hypot(Gradient_X, Gradient_Y)
    G = G / G.max() * 255  # Normalize to 255
    G = G.astype(np.uint8)  # Convert to 8-bit image for display

    theta = np.arctan2(Gradient_Y, Gradient_X)

    return G, theta

def final_sobel(img):
    if img is None:
        print("Error: Image not found or invalid image format.")
        return

    # Apply Sobel filters
    Gradient_X, Gradient_Y = apply_sobel_filters(img)

    # Compute magnitude and direction
    G, theta = compute_magnitude_and_direction(Gradient_X, Gradient_Y)


    return Gradient_X,Gradient_Y,G,theta
