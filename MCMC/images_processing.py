from PIL import Image
from matplotlib import image
from matplotlib import pyplot
import cv2
import numpy as np
import os

def dimensions(image):
    return image.shape


def efficient_way_to_read_images():
    data = image.imread('Images/bw_cat.png')  # a big matrix with zeros and ones. 0 is black, 1 corresponds to white
    # summarize shape of the pixel array
    print(data.dtype)
    print(data.shape)  # dimensionality
    # display the array of pixels as an image
    pyplot.imshow(data)
    pyplot.show()


# Convert a colored image to a monochrome image using Pillow library
def convert_to_monochrome(image_path):
    im = Image.open(image_path).convert('RGB')
    im = im.convert('1')  # convert image to black & white
    new_image_path = image_path[:-4] + '_bw' + image_path[-4:]  # add _bw suffix and save format
    im.save(new_image_path)
    # pixels = im.getdata()  # 0 is black, 255 is white
    # to-do: substitute 255 to 1


# Convert a colored image to black & white and greyscale images using OpenCV library. Show them in new windows and save to the working directory
def convert_to_bw_and_greyscale(original_image, filename, format):
    # original_image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    (thresh, black_and_white_image) = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
    im_titles = ('Black white image', 'Original image', 'Gray image')
    images = (black_and_white_image, original_image, gray_image)
    for title, image in zip(im_titles, images):
        open_image_in_window(title, image)
    # saving
    # filename = image_path[7:-4] # slice only image title (without 'Images' directory and the format)
    save_image(filename + '_bw', format, black_and_white_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def convert_to_bw_and_return(original_image):
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    (thresh, black_and_white_image) = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
    im_titles = ('Black white image', 'Original image', 'Gray image')
    images = (black_and_white_image, original_image, gray_image)
    for title, image in zip(im_titles, images):
        open_image_in_window(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return black_and_white_image


# Save image using OpenCV library
def save_image(filename, format, image, directory=''):
    #os.mkdir(directory)
    cv2.imwrite(os.path.join(directory, filename + '.' + format), image)


def open_image_in_window(title, image):
    cv2.imshow(title, image)


def resize(image, scale_percent):
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    new_dim = (width, height)
    resized_im = cv2.resize(image, new_dim, interpolation=cv2.INTER_AREA)
    return resized_im


if __name__ == '__main__':
    original_image = cv2.imread('../Images & results/Original images/brain7.jpg')
    #print(original_image.shape)
    bw_im = convert_to_bw_and_return(original_image)
    save_image('brain7_bw', 'jpg', bw_im, '../Images & results/Original binary images')


