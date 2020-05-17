import numpy as np
from skimage.metrics import structural_similarity
import cv2
import matplotlib.pyplot as plt

from math import log10, sqrt


## All these metrics must be symmetric (satisfy metrics axioms)
def ssi(imageX, imageY):
    """
        Calculate the Structural Similarity Index (SSIM) between the two images and show the difference image.
        The index takes values between -1 and 1, where 1 is a perfect similarity.

        :param imageX: the first image to compare
        :param imageY: the first second to compare
        :return: SSI coefficient
        """
    (score, diff) = structural_similarity(imageX, imageY, full=True, multichannel=True)
    diff = (diff * 255).astype("uint8")
    return (score, diff)


def mse(imageX, imageY):
    """
        Calculate the 'Mean Squared Error' (MSE) between two images (the sum of the squared differences between two pictures).
        0 means a perfect similarity. Images must have the same dimensions.

        :param imageX: the first image to compare
        :param imageY: the first second to compare
        :return: MSE coefficient (the larger, the more different images are)
        """
    squared_difference = (imageX.astype("float") - imageY.astype("float")) ** 2
    sum_of_sq_diff = np.sum(squared_difference)
    overall_pixels = imageX.shape[0] * imageX.shape[1]
    error = sum_of_sq_diff / overall_pixels  # divide the sum of squares by the overall number of pixels
    return error


# (1 - percentage of wrong pixels) can be treated as accuracy of samplers

# The idea is to minimize the number of wrong pixels
def percentage_of_wrong_pixels(imageX, imageY):
    """
    Perform pixel-wise comparison of two images. Can be used for evaluation of image restoration algorithms.

    :param imageX: the first image to compare
    :param imageY: the first second to compare
    :return: a percentage of wrong pixels
    """
    height = imageX.shape[0]
    width = imageX.shape[1]
    overall_pixels = width * height
    wrong_pixels = 0
    # iterate over all pixels and compare them. TODO: optimize for better performance
    for i in range(height):
        for j in range(width):
            if imageX[i, j].any() != imageY[i, j].any():
                wrong_pixels += 1
    return wrong_pixels / overall_pixels


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


def compare_with_wrong_pixels_metric(imageX, imageY, name):
    """Compare the given images using the value of the wrong pixels percentage loss function."""
    wrong_pixels_perc = percentage_of_wrong_pixels(imageX, imageY)
    figure = plt.figure(name)
    plt.suptitle("Percentage of wrong pixels: {0}".format(wrong_pixels_perc))
    # add the first image to the plot
    axis_1 = figure.add_subplot(1, 2, 1)
    plt.imshow(imageX, cmap=plt.cm.gray)
    plt.axis(False)
    # add the second image to the plot
    axis_2 = figure.add_subplot(1, 2, 2)
    plt.imshow(imageY, cmap=plt.cm.gray)
    plt.axis(False)
    plt.show()


# The higher the PSNR, the better the quality of the reconstructed image.
def PSNR(imageX, imageY):
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse(imageX, imageY)))
    return psnr


if __name__ == '__main__':
    original = cv2.imread('Binary images/bw_grumpy_cat.jpg')
    denoised = cv2.imread('Results/result_beta=0.8_noise_p=0.1_iter=1687500_neighbors=8.jpeg')
    #compare_with_metrics(original, denoised, 'Original vs Denoised images')
    compare_with_wrong_pixels_metric(original, denoised, 'Original vs Denoised images')
    noisy = cv2.imread('Noisy images/noised_10.0%_grumpy_cat.jpg')
    #compare_with_metrics(original, noisy, 'Original vs Noisy images')
    compare_with_wrong_pixels_metric(original, noisy, 'Original vs Noisy images')
    # score, diff = ssi(imageX, imageY)
    # print("SSIM: {}".format(score))
    # cv2.imshow("Difference", diff)
    # cv2.waitKey(0)
