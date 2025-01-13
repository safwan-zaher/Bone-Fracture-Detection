import os
import cv2
import numpy as np
import pandas as pd
import smoothing_filters as sf
import histogram as hist
import canny_edge as canny
import sobel_filter as sobel
import GLCM_features as glcm

def extract_features(img):
    # Apply median filter
    kernel_size = 5
    median_filtered = sf.median_filter(img, kernel_size)

    # Apply Gaussian filter
    height = 5
    width = 5
    sigmaX = 1.0
    sigmaY = 1.0
    kernel = sf.gaussian_kernel(height, width, sigmaX, sigmaY)
    gaussian_filtered = sf.convolution("Gaussian Filtered", kernel, median_filtered)

    # Apply histogram equalization
    _, CLAHE = hist.histogram(gaussian_filtered)

    # Apply Sobel filter
    _, _, G, theta = sobel.final_sobel(CLAHE)

    # Apply Canny edge detection
    non_max_img = canny.non_maximum_suppression(G, theta)
    threshold_img, weak, strong = canny.threshold(non_max_img)
    final_img = canny.hysteresis(threshold_img, weak, strong)

    # Extract GLCM features
    glcm_features = glcm.GLCM(final_img)

    return glcm_features

def load_path(path):
    dataset = []
    for body in os.listdir(path):
        body_part = body
        path_p = os.path.join(path, body)
        for id_p in os.listdir(path_p):
            patient_id = id_p
            path_id = os.path.join(path_p, id_p)
            for lab in os.listdir(path_id):
                if lab.split('_')[-1]=='positive':
                    label = 1
                elif lab.split('_')[-1]=='negative':
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


data_path = "C:/Users/USER/PycharmProjects/FinalProject/MURA-v1.1/train"
dataset = load_path(data_path)
feature_list = []
labels = []

for data in dataset:
    img_path = data['img_path']
    label = data['label']
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Image not found or invalid image format for {img_path}.")
        continue

    features = extract_features(img)
    feature_list.append(features)
    labels.append(label)

features_df = pd.DataFrame(feature_list)
labels_df = pd.DataFrame(labels, columns=['label'])
result_df = pd.concat([features_df, labels_df], axis=1)

result_df.to_csv("extracted_features.csv", index=False)
print("Feature extraction completed and saved to 'extracted_features.csv'")


