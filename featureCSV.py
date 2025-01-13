import os
import cv2
import numpy as np
import pandas as pd
import smoothing_filters as sf
import histogram as hist
import canny_edge as canny
import sobel_filter as sobel
import GLCM_features as glcm
from joblib import Parallel, delayed, cpu_count
from tqdm import tqdm

CHECKPOINT_FILE = "processed_images.txt"
INTERMEDIATE_CSV = "extracted_features_parallel.csv"


def extract_features(img):
    kernel_size = 5
    median_filtered = sf.median_filter(img, kernel_size)

    height = 5
    width = 5
    sigmaX = 1.0
    sigmaY = 1.0
    kernel = sf.gaussian_kernel(height, width, sigmaX, sigmaY)
    gaussian_filtered = sf.convolution("Gaussian Filtered", kernel, median_filtered)

    _, CLAHE = hist.histogram(gaussian_filtered)

    _, _, G, theta = sobel.final_sobel(CLAHE)

    non_max_img = canny.non_maximum_suppression(G, theta)
    threshold_img, weak, strong = canny.threshold(non_max_img)
    final_img = canny.hysteresis(threshold_img, weak, strong)

    glcm_features = glcm.GLCM(final_img)

    return glcm_features


def load_path(path):
    dataset = []
    for body in os.listdir(path):
        if body != "XR_HAND":  # Filter to only include hand images
            continue
        body_part = body
        path_p = os.path.join(path, body)
        for id_p in os.listdir(path_p):
            patient_id = id_p
            path_id = os.path.join(path_p, id_p)
            for lab in os.listdir(path_id):
                if lab.split('_')[-1] == 'positive':
                    label = 1
                elif lab.split('_')[-1] == 'negative':
                    label = 0
                path_l = os.path.join(path_id, lab)
                for img in os.listdir(path_l):
                    img_path = os.path.join(path_l, img)
                    dataset.append(
                        {
                            'body_part': body_part,
                            'patient_id': patient_id,
                            'label': label,
                            'img_path': img_path
                        }
                    )
    return dataset


def process_image(data):
    img_path = data['img_path']
    label = data['label']
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Image not found or invalid image format for {img_path}.")
        return None

    features = extract_features(img)
    return features, label, img_path, data['patient_id']


def save_intermediate_results(results, filename, checkpoint_file):
    processed_paths = []
    patient_ids = []
    labels = []
    correlations = []
    energies = []
    homogeneities = []
    contrasts = []
    dissimilarities = []

    for result in results:
        if result is not None:
            glcm_features = result[0]
            if glcm_features is None:
                print(f"Warning: GLCM features extraction failed for image {result[2]}. Skipping.")
                continue

            labels.append(result[1])
            processed_paths.append(result[2])
            patient_ids.append(result[3])
            correlations.append(glcm_features.get('Correlation', None))  # Accessing using .get() method
            energies.append(glcm_features.get('Energy', None))
            homogeneities.append(glcm_features.get('Homogeneity', None))
            contrasts.append(glcm_features.get('Contrast', None))
            dissimilarities.append(glcm_features.get('Dissimilarity', None))

    if labels:
        df_dict = {
            'patient_id': patient_ids,
            'label': labels,
            'img_path': processed_paths,
            'Correlation': correlations,
            'Energy': energies,
            'Homogeneity': homogeneities,
            'Contrast': contrasts,
            'Dissimilarity': dissimilarities
        }
        result_df = pd.DataFrame(df_dict)
        try:
            if not os.path.exists(filename):
                result_df.to_csv(filename, index=True, index_label="Index")
            else:
                result_df.to_csv(filename, mode='a', header=False, index=True, index_label="Index")
        except Exception as e:
            print(f"Error occurred while saving CSV file: {e}")

    with open(checkpoint_file, 'a') as f:
        for path in processed_paths:
            f.write(f"{path}\n")


def load_processed_images(checkpoint_file):
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            processed_images = set(f.read().splitlines())
    else:
        processed_images = set()
    return processed_images


if __name__ == '__main__':
    data_path = "C:/Users/USER/PycharmProjects/FinalProject/MURA-v1.1/train"
    dataset = load_path(data_path)
    processed_images = load_processed_images(CHECKPOINT_FILE)

    dataset = [data for data in dataset if data['img_path'] not in processed_images]

    num_processes = max(1, cpu_count() - 2)
    batch_size = 50

    for i in tqdm(range(0, len(dataset), batch_size)):
        batch = dataset[i:i + batch_size]
        results = Parallel(n_jobs=num_processes)(
            delayed(lambda data: process_image(data))(data) for data in batch
        )
        save_intermediate_results(results, INTERMEDIATE_CSV, CHECKPOINT_FILE)
        print(f"Batch {i // batch_size + 1} processed and saved.")

    print("Feature extraction completed in parallel and saved to 'extracted_features_parallel.csv'")
