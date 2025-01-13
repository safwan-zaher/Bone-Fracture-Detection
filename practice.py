import numpy as np


def compute_cooccurrence_matrix(img, max_intensity):

    # Initialize the co-occurrence matrix with zeros
    cooccurrence_matrix = np.zeros((max_intensity + 1, max_intensity + 1), dtype=np.int32)

    # Get the dimensions of the image
    rows, cols = img.shape

    # Iterate over each pixel in the image (except the last column)
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

    L = glcm.shape[0]  # Number of gray levels

    # Calculate mean for rows (μr) and columns (μc)
    mean_r = np.sum([i * np.sum(glcm[i, :]) for i in range(L)])
    mean_c = np.sum([j * np.sum(glcm[:, j]) for j in range(L)])

    # Calculate variance for rows (σ²r) and columns (σ²c)
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


# Example usage
if __name__ == "__main__":
    # Create the sample grayscale image
    img = np.array([
        [0, 0, 1, 1, 1],
        [0, 0, 1, 1, 1],
        [0, 2, 2, 2, 2],
        [2, 2, 3, 3, 3],
        [2, 2, 3, 3, 3]
    ], dtype=np.int32)

    max_intensity = np.max(img)

    cooccurrence_matrix = compute_cooccurrence_matrix(img, max_intensity)
    print("Co-occurrence Matrix:")
    print(cooccurrence_matrix)

    normalized_matrix = normalize_cooccurrence_matrix(cooccurrence_matrix)
    print("Normalized Symmetrical GLCM Matrix:")
    print(normalized_matrix)

    mean_r, mean_c, var_r, var_c = compute_mean_variance(normalized_matrix)
    print(f"Mean (Rows): {mean_r}, Mean (Columns): {mean_c}")
    print(f"Variance (Rows): {var_r}, Variance (Columns): {var_c}")

    correlation = compute_correlation(normalized_matrix, mean_r, mean_c, var_r, var_c)
    print(f"Correlation: {correlation}")
