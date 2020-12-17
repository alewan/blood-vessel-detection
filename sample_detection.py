# Authored by Aleksei Wan on December 16, 2020
# Performs the detection and shows images for demonstration purposes
import cv2
import numpy as np

# This is coded separately from the function from the other file because adding a conditional regarding plotting
# logic would slow down the code, which would not be ideal for this use case.

# Parameter
THRESHOLD = 100
IMAGE_FILE_NAME = 'test.jpg'

# Read in image and transform it
im = cv2.imread(IMAGE_FILE_NAME)
cv2.imshow('Original', im)
im_grey = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
cv2.imshow('Grey', im_grey)

# Inverting the image to highlight the areas of interest since they are dark
# Note: This is not actually required/used by Canny
im_mod = cv2.bitwise_not(im_grey)
cv2.imshow('Inverted', im_mod)

# Detect edges using Canny
# Typically you would blur (i.e. im_mod = cv2.blur(im_mod, (3,3))), but in this case it performs worse
# This is taken from the OpenCV demo https://docs.opencv.org/3.4/df/d0d/tutorial_find_contours.html
canny_output = cv2.Canny(im_grey, THRESHOLD, THRESHOLD * 2)

# Find all shapes
_, contours, hierarchy = cv2.findContours(canny_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Draw the shapes on the images for reference
drawing = np.zeros((im_grey.shape[0], im_grey.shape[1], 3), dtype=np.uint8)
color = (0, 0, 255)
for i in range(len(contours)):
    cv2.drawContours(drawing, contours, i, color, 2, cv2.LINE_8, hierarchy, 0)
    cv2.drawContours(im, contours, i, color, 2, cv2.LINE_8, hierarchy, 0)
cv2.imshow('Detected', drawing)
cv2.imshow('Detected Superimposed on Original', im)

# Print the number of shapes it found
print('Num Detected Shapes:', len(contours))

cv2.waitKey(0)
cv2.destroyAllWindows()
