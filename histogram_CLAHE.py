import numpy as np
import matplotlib.pyplot as plt
import cv2

def clip_histogram(hist, clip_limit):
    # Clip histogram based on clip_limit
    excess = np.maximum(hist - clip_limit, 0)
    clipped_hist = np.minimum(hist, clip_limit)
    return clipped_hist, excess.sum()

def redistribute_excess(hist, excess):
    # Redistribute excess pixels in histogram
    hist += excess // 256
    hist[:excess % 256] += 1
    return hist

def clahe(image, clip_limit=2.0):
    img_shape = image.shape
    img_clahe = np.zeros(img_shape, dtype=np.uint8)

    # Compute histogram
    hist, _ = np.histogram(image.flatten(), bins=256, range=(0, 256))
    clip_limit_value = clip_limit * (img_shape[0] * img_shape[1]) / 256

    # Adjust clip limit dynamically based on histogram statistics
    clip_limit_value = min(clip_limit_value, np.max(hist))

    # Clip histogram and redistribute excess
    clipped_hist, total_excess = clip_histogram(hist, clip_limit_value)
    redistributed_hist = redistribute_excess(clipped_hist, int(total_excess))

    # Compute cumulative distribution function (CDF)
    cdf = np.cumsum(redistributed_hist).astype(np.float32)
    cdf_normalized = 255 * cdf / cdf[-1]

    # Apply CDF mapping to the entire image
    img_clahe = cdf_normalized[image].astype(np.uint8)

    return img_clahe

def display_images(original, processed):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(original, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title('Processed Image')
    plt.imshow(processed, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

# Load the image
original_image = cv2.imread('broken.jpg', cv2.IMREAD_GRAYSCALE)

# Ensure the image is loaded
if original_image is None:
    raise ValueError("Image not found.")
else:
    print(f"Image loaded successfully with shape: {original_image.shape}")

# Apply CLAHE
clahe_image = clahe(original_image, clip_limit=3.0)  # Adjust clip_limit as needed

# Display the images
display_images(original_image, clahe_image)
