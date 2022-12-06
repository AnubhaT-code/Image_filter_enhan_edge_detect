import cv2
import math
import numpy as np
import sys
from PIL import Image
from PIL import ImageEnhance
import os

def grayscale(image):
    # converting image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
    #cv2.imshow('Grayscale', gray_image)
    cv2.waitKey(0)
    return gray_image


def histequal(gray_image):
    # creating a Histogram Equalization of a image using cv2.equalizeHist()
    equ = cv2.equalizeHist(gray_image)
    # cv2.imshow("image",equ)
    cv2.waitKey(0)
    return equ

def gamma(gray_image,a,c):
    image = gray_image.copy()
    image = Image.fromarray(image)
    image = image.convert('RGB')
    width , height = image.size
    for x in range(0,width):
        for y in range(0,height):
            r, g, b = image.getpixel((x,y))
            value = (int(c*math.pow(r,a)),int(c*math.pow(g,a)),int(c*math.pow(b,a)))
            image.putpixel((x, y), value)
    return image
#cv2.imshow("newimage",image)

# Enhance Brightness

def imageEnhance(image,brightness_change,sharpness_change,contrast_change):

    brightness = ImageEnhance.Brightness(image)
    image = brightness.enhance(brightness_change)
    
    sharpness = ImageEnhance.Sharpness(image)
    image = sharpness.enhance(sharpness_change)

    contrast = ImageEnhance.Contrast(image)
    image = contrast.enhance(contrast_change)

    return image

# read an image using imread
n_input = sys.argv[1]
image = cv2.imread(n_input,cv2.IMREAD_COLOR)
# cv2.imshow("input image",img)

filename = os.path.basename(n_input).split('/')[-1]

if filename=="cctv1.jpg":
    image = grayscale(image)
    image = gamma(image,1.8,1)  
    image= imageEnhance(image,1.2,8.3,0.5)

    # image = np.array(image)
    # image = cv2.GaussianBlur(image, (5, 5), 0)
    # canny = cv2.Canny(image,20,100,apertureSize = 3)
    # image = cv2.subtract(grayscale(image),canny)
    # image = Image.fromarray(image)

    image.save(f"enhanced-{filename}")

if filename=="cctv2.jpg":
    image = grayscale(image)
    image2 = histequal(image)

    image1 = cv2.GaussianBlur(image, (181, 181), 41)
    image3 = cv2.subtract(image,image1)
    

    image = gamma(image,2.0,1)  
    image= imageEnhance(image,1.2,7.8,0.6)
    image.save(f"enhanced-{filename}")

if filename=="cctv3.jpg":
    image = grayscale(image)
    image = gamma(image,2.0,1)  
    image= imageEnhance(image,1.2,5.3,0.5)
    image.save(f"enhanced-{filename}")

if filename=="cctv4.jpg":
    image = grayscale(image)
    image = gamma(image,1.5,1)  
    image= imageEnhance(image,1.0,18.3,0.8)
    image.save(f"enhanced-{filename}")

cv2.destroyAllWindows()