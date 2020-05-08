import numpy as np
from skimage.metrics import structural_similarity
import argparse
import imutils
import cv2
import matplotlib.pyplot as plt


# For reading arguments passed from the command line
# arg_parser = argparse.ArgumentParser()

# arg_parser.add_argument("-f", "--first", required=True, help="the first input image")
# arg_parser.add_argument("-s", "--second", required=True, help="the second input image")
# args = vars(arg_parser.parse_args())

# imageA = cv2.imread(args["first"])
# imageB = cv2.imread(args["second"])

# TODO: pairwise pixel similarity

# Acccuracy


def ssi(imageX, imageY):
    """Calculate the Structural Similarity Index (SSIM) between the two images and show the difference image.
     The index takes values between -1 and 1, where 1 is a perfect similarity.
    """
    (score, diff) = structural_similarity(imageX, imageY, full=True, multichannel=True)
    diff = (diff * 255).astype("uint8")
    return (score, diff)


# MSE is the sum of the squared differences between two pictures
def mse(imageX, imageY):
    """Calculate the 'Mean Squared Error' (MSE) between two images. 0 means a perfect similarity. Images must have the same dimensions."""
    error = np.sum((imageX.astype("float") - imageY.astype("float")) ** 2)
    squared_difference = (imageX.astype("float") - imageY.astype("float"))** 2
    sum_of_sq_diff = np.sum(squared_difference)
    overall_pixels = imageX.shape[0] * imageX.shape[1]
    error = sum_of_sq_diff / overall_pixels  # divide the sum of squares by the overall number of pixels
    return error
# Perform pixel-wise comparison
def percentage_of_wrong_pixels():
    pass
def compare_with_metrics(imageX, imageY, name):
    """Compare the given images using the MSE and SSI metrics."""
    mean_sq_err = mse(imageX, imageY)
    structural_sim_score, diff_image = ssi(imageX, imageY)
    figure = plt.figure(name)
    plt.suptitle("SSIM: {0}, MSE: {1}".format(structural_sim_score, mean_sq_err))
    # add the first image to the plot
    axis_1 = figure.add_subplot(1, 2, 1)
    plt.imshow(imageX, cmap=plt.cm.gray)
    plt.axis(False)
    # add the second image to the plot
    axis_2 = figure.add_subplot(1, 2, 2)
    plt.imshow(imageY, cmap=plt.cm.gray)
    plt.axis(False)
    plt.show()


if __name__ == '__main__':
    # Read directly from the file system
    imageX = cv2.imread('Images/cat_bw.png')
    imageY = cv2.imread('AfterSampling2000000.png')
    compare_with_metrics(imageX, imageY, 'Original vs Denoised')
    # score, diff = ssi(imageX, imageY)
    # print("SSIM: {}".format(score))
    # cv2.imshow("Difference", diff)
    # cv2.waitKey(0)
