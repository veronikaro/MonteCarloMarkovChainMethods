from PIL import Image
from matplotlib import image
from matplotlib import pyplot
import cv2
import numpy as np


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


# Convert a colored image to black & white and greyscale images using OpenCV library. Show them in new windows
def convert_to_bw_and_greyscale(image_path):
    original_image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    (thresh, black_and_white_image) = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
    im_titles = ('Black white image', 'Original image', 'Gray image')
    images = (black_and_white_image, original_image, gray_image)
    for title, image in zip(im_titles, images):
        open_image_in_window(title, image)
    # saving
    filename = image_path[7:-4] # slice only image title (without 'Images' directory and the format)
    save_image(filename + '_bw', black_and_white_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Save image using OpenCV library
def save_image(filename, image):
    cv2.imwrite('Images/' + filename + '.png', image)

def open_image_in_window(title, image):
    cv2.imshow(title, image)

if __name__ == '__main__':
    # efficient_way_to_read_images()
    convert_to_bw_and_greyscale('Images/cat.jpg')


def convert_to_monochrome():
    '''
    im = Image.open('cat.jpg').convert('RGB')
    im = im.convert('1')  # convert image to black and white
    im.save('bw_cat.png')
    '''
    im = Image.open('bw_cat.png')
    pixels = im.getdata()  # 0 is black, 255 is white
    # to-do: substitute 255 to 1

    # write pixels into a matrix that will be comfortable to work with
    for i, pix in zip(range(len(pixels)), pixels):
        # print(i, pix)
        if pix == 255:
            pixels[i] = 1  # exception!
    for i, pix in zip(range(len(pixels)), pixels):
        print(i, pix)
        if i == 20:
            break
