#Image Denoising Using Median Filter

import matplotlib.pyplot as plt
import urllib.request
from skimage import io, img_as_float
from skimage.filters import median
from skimage.morphology import disk

# Define the URL of the image to be processed 
# Take image of your choice. This is a sample. 
image_url = 'https://cdn.esahubble.org/archives/images/wallpaper2/heic1307a.jpg'

# Download the image from the URL
urllib.request.urlretrieve(image_url, 'input_image.jpg')

# Read the image in grayscale
img_gray = img_as_float(io.imread('input_image.jpg', as_gray=True))

# Apply median filter to the grayscale image
median_using_skimage = median(img_gray, disk(4), mode='constant')

# Plot the original and denoised images
plt.figure(figsize=(15, 15))

# Original Image
plt.subplot(1, 2, 1)
plt.imshow(img_gray, cmap='gray')
plt.title('Original Image')
plt.axis('off')

# Denoised Image
plt.subplot(1, 2, 2)
plt.imshow(median_using_skimage, cmap='gray')
plt.title('Image Denoising Using Median Filter')
plt.axis('off')

# parallel plots
plt.show()
