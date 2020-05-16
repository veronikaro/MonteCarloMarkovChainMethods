from MCMC.services.noiser import sp_noise
from PIL import Image
from matplotlib import image
from matplotlib import pyplot
import cv2
from os import path, mkdir
import sys
from imghdr import what


def dimensions(image):
    return image.shape


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
def save_image(filename, image, directory=''):
    if not path.exists(directory):
        mkdir(directory)
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
    save_image('bw_{0}'.format(original_image_name), bw_image, 'Binary images')
    noised_bw_image = sp_noise(bw_image, noise_probability)
    # save the noised version of the black & white image
    save_image('noised_{0}'.format(original_image_name), noised_bw_image, 'Noisy images')


if __name__ == '__main__':
    arguments = sys.argv
    original_image_name, noise_probability = arguments[1:]
    prepare_image_for_sampler_pipeline(original_image_name, float(noise_probability))
    print('success')
