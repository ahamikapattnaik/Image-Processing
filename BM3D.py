#Install BM3D

!pip install bm3d

#BM3D

import matplotlib.pyplot as plt
from skimage import io, img_as_float
import bm3d
import cv2
import urllib.request

#Sample Image

image_url = 'https://cdn.esahubble.org/archives/images/wallpaper2/heic1307a.jpg'
urllib.request.urlretrieve(image_url, 'input_image.jpg')
img_gray = img_as_float(io.imread("input_image.jpg", as_gray=True))
BM3D_denoised_image = bm3d.bm3d(img_gray, sigma_psd=0.1, stage_arg=bm3d.BM3DStages.HARD_THRESHOLDING)
plt.figure(figsize=(12,8))
plt.subplot(1,2,1)
plt.imshow(img_gray)
plt.title("Original Image")
plt.axis('off')
plt.subplot(1,2,2)
plt.imshow(BM3D_denoised_image)
plt.title('BM3D Denoised Image')
plt.axis('off')
