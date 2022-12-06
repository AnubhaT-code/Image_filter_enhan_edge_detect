import cv2
import numpy as np
import sys
import os
from PIL import Image

n_input = sys.argv[1] 
img = cv2.imread(n_input,cv2.IMREAD_COLOR)

# Grayscale image
gray1 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# cv2.imshow('gray',gray1)
cv2.waitKey(0) 

# Gaussian Blur
gray = cv2.GaussianBlur(gray1, (5, 5), 0)
# cv2.imshow('Gaussian Blurring', gray)
cv2.waitKey(0)

# Canny edge detector
edges = cv2.Canny(gray,20,100,apertureSize = 3)
# cv2.imshow('canny',edges)
cv2.waitKey(0)

kernel = np.ones((1,1),np.uint8)
edges = cv2.dilate(edges,kernel,iterations = 3)
# cv2.imshow('dilate',edges)
cv2.waitKey(0)

kernel = np.ones((1,1),np.uint8)
edges = cv2.erode(edges,kernel,iterations = 1)
# cv2.imshow('erode',edges)
cv2.waitKey(0)

lines = cv2.HoughLines(edges,1,np.pi/180,150)
print(" Total Hough lines are ", lines.size//2)

for line in lines:
    rho,theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))

    cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)

# cv2.imshow('hough.jpg',img)
img = Image.fromarray(img)
filename = os.path.basename(n_input).split('/')[-1]
img.save(f"robolin-{filename}")
cv2.waitKey(0)
cv2.destroyAllWindows()