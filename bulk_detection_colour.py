# Authored by Aleksei Wan in September 2021
# Performs the detection and prints the number of contours in each image
import os
import re
import cv2
import numpy as np
from scipy import spatial
import pandas as pd
from argparse import ArgumentParser

# regex for image file matching
IMG_FILE = re.compile('(.*)\.jp[e]?g$')


# Perform Detection
def detect_on_image_with_colour(image: str, threshold: int, colour: np.ndarray) -> int:
    """
    Perform detection on provided image weighted by colour using cosine similarity
    :param image: path to image to perform detection on
    :param threshold: threshold to use for Canny detection
    :param colour: BGR colour to use
    :return: number of shapes in the image
    """
    im = cv2.imread(image)
    im_grey_inv = cv2.bitwise_not(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY))
    similarity = np.zeros([np.size(im, 0), np.size(im, 1)])
    imtemp = im.astype(np.float64)
    for x in range(np.shape(im)[0]):
        for y in range(np.shape(im)[1]):
            similarity[x, y] = 1 - spatial.distance.cosine(imtemp[x, y], colour)

    similarity -= np.min(similarity)
    similarity /= np.max(similarity)
    brown_sim_im = im_grey_inv.astype(np.float64) * similarity
    canny_output = cv2.Canny(brown_sim_im.astype(np.uint8), threshold, threshold * 2)

    # Note: other versions of opencv may require `_, contours, hierarchy = ...`
    contours, hierarchy = cv2.findContours(canny_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    return len(contours)


if __name__ == "__main__":
    parser = ArgumentParser(description='Perform detection on all images in the directory')
    parser.add_argument('--input_dir', type=str, default='images', help='Directory containing images')
    parser.add_argument('--red', '-r', type=int, default=153, help='Red value of desired colour')
    parser.add_argument('--green', '-g', type=int, default=0, help='Red value of desired colour')
    parser.add_argument('--blue', '-b', type=int, default=76, help='Red value of desired colour')
    parser.add_argument('--threshold', type=int, default=100, help='Detection threshold')
    parser.add_argument('--output', type=str, default='results.csv', help='Location for the results')
    args = parser.parse_args()

    path_to_check = os.path.abspath(args.input_dir)

    if not os.path.exists(path_to_check):
        print('Provided path', path_to_check, 'is not a valid directory. Please try again.')
        exit(-1)

    if not ((0 < args.red < 255) and (0 < args.green < 255) and (0 < args.blue < 255)):
        print('Provided RGB values invalid.')
        exit(-1)

    images = os.listdir(path_to_check)
    imgs, results = [], []

    tracking, num_files = 0, str(len(images))
    for img in images:
        tracking += 1
        print('(' + str(tracking) + '/' + num_files + ')', end=' ')
        img_file = re.match(IMG_FILE, img)
        if img_file:
            print('Evaluating image', img_file[1])
            img_path = os.path.join(path_to_check, img)
            imgs.append(img_file[1])
            results.append(detect_on_image_with_colour(img_path, args.threshold, np.array([args.blue, args.green, args.red])))
        else:
            print('Ignoring non-image file', img)

    results = pd.DataFrame(data={'Image Name': imgs, 'Shapes Detected': results})
    results.to_csv(path_or_buf=args.output, index=False)
