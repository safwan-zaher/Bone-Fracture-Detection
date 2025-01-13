import cv2
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
import smoothing_filters as sf
import histogram as hist
import canny_edge as canny
import sobel_filter as sobel
import GLCM_features as glcm
from tkinter import Tk, filedialog, Button, Label, Toplevel, StringVar

rf_model = joblib.load('rf_model.pkl')

scaler = joblib.load('scaler.pkl')

# Define a function to extract features from an image
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

def predict_image(img_path):

    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)


    features = extract_features(image)


    feature_df = pd.DataFrame([features], columns=['Correlation', 'Energy', 'Homogeneity', 'Contrast', 'Dissimilarity'])
    features_normalized = scaler.transform(feature_df)
    rf_prediction = rf_model.predict(features_normalized)

    return rf_prediction[0]

def open_file():
    # Open file dialog to choose an image
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
    if file_path:
        # Show loading sign
        loading_window = Toplevel(root)
        loading_window.title("Loading")
        loading_label = Label(loading_window, text="Processing...", padx=20, pady=20)
        loading_label.pack()
        root.update_idletasks()

        rf_pred = predict_image(file_path)
        rf_result = 'Fractured' if rf_pred == 1 else 'Not Fractured'
        result_label.config(text=f"Random Forest Prediction: {rf_result}")

        loading_window.destroy()


root = Tk()
root.title("X-Ray Image Classifier")


open_button = Button(root, text="Open Image", command=open_file)
open_button.pack(pady=20)

result_label = Label(root, text="Select an image to classify", padx=20, pady=20)
result_label.pack(pady=20)


root.mainloop()
