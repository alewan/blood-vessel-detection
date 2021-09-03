# Authored by Aleksi Wan & Alex Mertens 2020-2021
# Performs the detection and shows images for demonstration purposes
import cv2
import numpy as np
from scipy import spatial

# This is coded separately from the function from the other file because adding a conditional regarding plotting
# logic would slow down the code, which would not be ideal for this use case.

# Parameter
THRESHOLD = 75
IMAGE_FILE_NAME = 'test.jpg'
HIGHLIGHT_COLOUR = (0, 0, 255)
DESIRED_COLOUR = np.array([76, 0, 153])  # Colour value in BGR, brown is 0, 56, 173

# Read in image and transform it
im = cv2.imread(IMAGE_FILE_NAME)
similarity = np.zeros([np.size(im, 0), np.size(im, 1)])
cv2.imshow('Original', im)

imtemp = im.astype(np.float64)
for x in range(np.shape(im)[0]):
    for y in range(np.shape(im)[1]):
        similarity[x, y] = 1 - spatial.distance.cosine(imtemp[x, y], DESIRED_COLOUR)

similarity -= np.min(similarity)
similarity /= np.max(similarity)

im_grey = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

# Inverting the image to highlight the areas of interest since they are dark
# Note: This is not actually required/used by Canny
im_mod = cv2.bitwise_not(im_grey)

bsim = im_mod.astype(np.float64) * similarity
bsim = bsim.astype(np.uint8)

cv2.imshow('Inverted', im_mod)
cv2.imshow('Brown Similar inverted', bsim)

# Detect edges using Canny
# Typically you would blur (i.e. im_mod = cv2.blur(im_mod, (3,3))), but in this case it performs worse
# This is taken from the OpenCV demo https://docs.opencv.org/3.4/df/d0d/tutorial_find_contours.html
canny_output = cv2.Canny(im_grey, THRESHOLD, THRESHOLD * 2)
canny_output2 = cv2.Canny(bsim, THRESHOLD, THRESHOLD * 2)

# Find all shapes
contours, hierarchy = cv2.findContours(canny_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours2, hierarchy2 = cv2.findContours(canny_output2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Draw the shapes on the images for reference
drawing = np.zeros((im_grey.shape[0], im_grey.shape[1], 3), dtype=np.uint8)
drawing2 = np.zeros((im_grey.shape[0], im_grey.shape[1], 3), dtype=np.uint8)

im2 = im.copy()
for i in range(len(contours)):
    cv2.drawContours(drawing, contours, i, HIGHLIGHT_COLOUR, 2, cv2.LINE_8, hierarchy, 0)
    cv2.drawContours(im, contours, i, HIGHLIGHT_COLOUR, 2, cv2.LINE_8, hierarchy, 0)

for i in range(len(contours2)):
    cv2.drawContours(drawing2, contours2, i, HIGHLIGHT_COLOUR, 2, cv2.LINE_8, hierarchy2, 0)
    cv2.drawContours(im2, contours2, i, HIGHLIGHT_COLOUR, 2, cv2.LINE_8, hierarchy2, 0)

cv2.imshow('Detected', drawing)
cv2.imshow('Detected2', drawing2)

cv2.imshow('Detected Superimposed on Original', im)
cv2.imshow('Detected2 Superimposed on Original', im2)

# Print the number of shapes it found
print('Num Detected Shapes:', len(contours))
print('Num Detected Shapes2:', len(contours2))

cv2.waitKey(0)
cv2.destroyAllWindows()
