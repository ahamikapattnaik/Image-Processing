# Create bilateral filter from scratch using NumPy

import cv2
import numpy as np
import urllib.request
from matplotlib import pyplot as plt

def bilateral_filter(image, sigma_s, sigma_r):
    image = image.astype(np.float32)
    filtered_image = np.zeros_like(image)

    #defining a spatial Gaussian kernel
    spatial_kernel_size = 2 * sigma_s + 1
    spatial_kernel = np.fromfunction(lambda x, y: np.exp(-((x - sigma_s)**2 + (y - sigma_s)**2) / (2 * sigma_s**2)),
                                      (spatial_kernel_size, spatial_kernel_size))

    #iterating over each pixel
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            #neighborhood of pixel
            i_min, i_max = max(0, i - sigma_s), min(image.shape[0], i + sigma_s + 1)
            j_min, j_max = max(0, j - sigma_s), min(image.shape[1], j + sigma_s + 1)
            neighborhood = image[i_min:i_max, j_min:j_max]

            range_kernel = np.exp(-((neighborhood - image[i, j])**2) / (2 * sigma_r**2))

            weighted_sum = np.sum(range_kernel * spatial_kernel[:i_max-i_min, :j_max-j_min] * neighborhood)
            normalization_factor = np.sum(range_kernel * spatial_kernel[:i_max-i_min, :j_max-j_min])
            filtered_image[i, j] = weighted_sum / normalization_factor

    return filtered_image.astype(np.uint8)


image_url = 'https://cdn.esahubble.org/archives/images/wallpaper2/heic1307a.jpg'
urllib.request.urlretrieve(image_url, 'input_image.jpg')
image = cv2.imread('input_image.jpg', cv2.IMREAD_GRAYSCALE)
sigma_s = 3
sigma_r = 60
filtered_image = bilateral_filter(image, sigma_s, sigma_r)

#parallel subplot

plt.figure(figsize=(12,8))
plt.subplot(1,2,1)
plt.title('Original Image')
plt.axis('off')
plt.imshow(image,cmap='gray')
plt.subplot(1,2,2)
plt.imshow(filtered_image,cmap='gray')
plt.title('Bilateral Filtered Image')
plt.axis('off')
plt.show()
