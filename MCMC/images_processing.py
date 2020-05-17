from MCMC.services.noiser import sp_noise
from matplotlib import image
from matplotlib import pyplot
import cv2
from os import path, mkdir
import sys
from imghdr import what
import numpy as np


def dimensions(image):
    return image.shape


# TODO: think how to improve for better performance when iterating over pixels.
def reduce_channels_for_sampler(image):
    """Reduce a 3-channel image to 1-channel image."""
    w = image.shape[0]  # switch width to height
    h = image.shape[1]
    new_image = np.ndarray(shape=(w, h))  # create a new array for pixel values
    for i in range(w):  # rows
        for j in range(h):  # columns
            new_image[i][j] = image[i, j][:1]
    return new_image


# TODO: move to the images_processing module
def restore_channels(image, n_channels):
    """
    Convert 1-channel image to n-channel image.

    :param image: an image in ndarray format
    :param n_channels: target number of channels
    :return: ndarray with image's pixels' intensities
    """
    width, height, m = image.shape[0], image.shape[1], 1
    arr = image.reshape(width, height)
    return np.repeat(arr, m * n_channels).reshape(width, height, n_channels)


def convert_image_to_ising_model(image):
    """Convert the image to the Ising model representation (with a spin space = {-1, 1})."""
    image[np.where(image == [0])] = [-1]
    image[np.where(image == [255])] = [1]
    return image

def convert_from_ising_to_image(image):
    """Convert the Ising representation of an image to the standard image format with pixels' intensities."""
    image[np.where(image == [-1])] = [0]
    image[np.where(image == [1])] = [255]
    return image


# for experimenting
def efficient_way_to_read_images():
    data = image.imread('Images/bw_cat.png')  # a big matrix with zeros and ones. 0 is black, 1 corresponds to white
    # summarize shape of the pixel array
    print(data.dtype)
    print(data.shape)  # dimensionality
    # display the array of pixels as an image
    pyplot.imshow(data)
    pyplot.show()


# Convert a colored image to black & white and greyscale images using OpenCV library. Show them in new windows and save to the working directory
def convert_to_bw_and_greyscale_and_show(image_path, filename):
    original_image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    (thresh, black_and_white_image) = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
    im_titles = ('Black white image', 'Original image', 'Gray image')
    images = (black_and_white_image, original_image, gray_image)
    for title, image in zip(im_titles, images):
        open_image_in_window(title, image)
    format = what(image_path)
    save_image(filename + '_bw', format, black_and_white_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def convert_to_bw(original_image):
    """
    Return an image converted to black & white (binary) format.
    :param original_image: an arbitrary image sized mxn
    :return: the b&w image
    """
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    (thresh, black_and_white_image) = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
    return black_and_white_image


# Save image using the OpenCV library
def save_image(filename, format, image, directory=''):
    if not path.exists(directory):
        mkdir(directory)
    filename = '{0}.{1}'.format(filename, format)
    cv2.imwrite(path.join(directory, filename), image)


def open_image_in_window(title, image):
    cv2.imshow(title, image)


def resize(image, scale_percent):
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    new_dim = (width, height)
    resized_im = cv2.resize(image, new_dim, interpolation=cv2.INTER_AREA)
    return resized_im


# Run this script from the command line to get two versions of an image:
# 1. Black & white
# 2. Noisy
def prepare_image_for_sampler_pipeline(original_image_name, noise_probability):
    original_image = cv2.imread(original_image_name)
    bw_image = convert_to_bw(original_image)
    # save the black & white version of the image
    format = what(original_image_name)
    filename = path.splitext(original_image_name)[0] # without extension
    save_image('bw_{0}'.format(filename), format, bw_image, 'Binary images')
    noised_bw_image = sp_noise(bw_image, noise_probability)
    # save the noised version of the black & white image
    noised_pixels_percentage = noise_probability * 100
    save_image('noised_{0}%_{1}'.format(noised_pixels_percentage, filename), format, noised_bw_image, 'Noisy images')


if __name__ == '__main__':
    arguments = sys.argv
    original_image_name, noise_probability = arguments[1:]
    prepare_image_for_sampler_pipeline(original_image_name, float(noise_probability))
    print('successfully converted to b&w and added some noise')
