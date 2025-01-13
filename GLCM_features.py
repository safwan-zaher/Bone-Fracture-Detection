import numpy as np


def compute_cooccurrence_matrix(img, max_intensity):
    cooccurrence_matrix = np.zeros((max_intensity + 1, max_intensity + 1), dtype=np.int32)
    rows, cols = img.shape

    for i in range(rows):
        for j in range(cols - 1):  # Avoid the last column
            current_pixel = img[i, j]
            right_pixel = img[i, j + 1]
            cooccurrence_matrix[current_pixel, right_pixel] += 1

    return cooccurrence_matrix

def normalize_cooccurrence_matrix(cooccurrence_matrix):
    total_occurrences = np.sum(cooccurrence_matrix)
    normalized_matrix = cooccurrence_matrix / total_occurrences
    return normalized_matrix


def compute_mean_variance(glcm):
    L = glcm.shape[0]
    mean_r = np.sum([i * np.sum(glcm[i, :]) for i in range(L)])
    mean_c = np.sum([j * np.sum(glcm[:, j]) for j in range(L)])

    var_r = np.sum([(i - mean_r) ** 2 * np.sum(glcm[i, :]) for i in range(L)])
    var_c = np.sum([(j - mean_c) ** 2 * np.sum(glcm[:, j]) for j in range(L)])
    return mean_r, mean_c, var_r, var_c


def compute_correlation(glcm, mean_r, mean_c, var_r, var_c):
    L = glcm.shape[0]
    std_r = np.sqrt(var_r)
    std_c = np.sqrt(var_c)
    correlation = np.sum([[(i - mean_r) * (j - mean_c) * glcm[i, j] for j in range(L)] for i in range(L)]) / (
                std_r * std_c)
    return correlation


def compute_energy(glcm):
    energy = np.sum(glcm ** 2)
    return energy


def compute_homogeneity(glcm):
    L = glcm.shape [0]
    homogeneity = np.sum([[(glcm[i, j] / (1 + np.abs(i - j))) for j in range(L)] for i in range(L)])
    return homogeneity


def compute_contrast(glcm):
    L = glcm.shape[0]
    contrast = np.sum([[(i - j) ** 2 * glcm[i, j] for j in range(L)] for i in range(L)])
    return contrast


def compute_dissimilarity(glcm):
    L = glcm.shape[0]
    dissimilarity = np.sum([[np.abs(i - j) * glcm[i, j] for j in range(L)] for i in range(L)])
    return dissimilarity

def GLCM(img):
    max_intensity = np.max(img)

    cooccurrence_matrix = compute_cooccurrence_matrix(img, max_intensity)
    normalized_matrix = normalize_cooccurrence_matrix(cooccurrence_matrix)

    mean_r, mean_c, var_r, var_c = compute_mean_variance(normalized_matrix)

    correlation = compute_correlation(normalized_matrix, mean_r, mean_c, var_r, var_c)
    energy = compute_energy(normalized_matrix)
    homogeneity = compute_homogeneity(normalized_matrix)
    contrast = compute_contrast(normalized_matrix)
    dissimilarity = compute_dissimilarity(normalized_matrix)

    return {
        'Correlation': correlation,
        'Energy': energy,
        'Homogeneity': homogeneity,
        'Contrast': contrast,
        'Dissimilarity': dissimilarity
    }


