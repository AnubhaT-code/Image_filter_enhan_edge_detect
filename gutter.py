import numpy as np
import cv2
from PIL import Image
import sys

n_input = sys.argv[1]
# taking input from linux command
image = cv2.imread(n_input)

#cv2.imshow('Original Image', image)
cv2.waitKey(0)

# converting image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
 
#cv2.imshow('Grayscale', gray_image)
cv2.waitKey(0)

# Gaussian Blur
# with kernal size of 81 X 81 and standard deviation of 41
Gaussian = cv2.GaussianBlur(gray_image, (81, 81), 41)
#cv2.imshow('Gaussian Blurring', Gaussian)
cv2.waitKey(0)

# subtract the images
#image7 = np.array(image)

subtracted1 = cv2.subtract(Gaussian,gray_image)
subtracted1 = cv2.subtract(255,subtracted1)

# TO show the output
#cv2.imshow('image1', subtracted1)
subtracted1 = Image.fromarray(subtracted1)
subtracted1.save("cleaned-gutter.jpg")
cv2.waitKey(0)