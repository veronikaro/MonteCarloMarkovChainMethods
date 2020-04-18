import numpy as np
import timeit
import random
import cython
from math import exp, log
from MCMC.images_processing import *
from itertools import repeat
import cv2


# TO-DO: describe all parameters used in the algorithm

def convert_image_to_ising_model(image):
    converted = []
    image[np.where(image == [0])] = [-1]
    image[np.where(image == [255])] = [1]
    '''
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i][j] == 0:
                image[i][j] = -1
            elif image[i][j] == 255:
                image[i][j] = 1
    '''
    return image

def convert_from_ising_to_image(image):
    image[np.where(image == [-1])] = [0]
    image[np.where(image == [1])] = [255]
    return image

# Calculate an acceptance probability of flipping the pixel or not.
# Accepts a monochrome image with pixel intensities converted to -1 and +1
def clique_energy(image, pixel_position):
    neighbors = get_all_neighbors(pixel_position, image.shape)
    pixel_intensities = []
    for n in neighbors:
        pixel_intensities.append(image[n])
    return sum(pixel_intensities)


def acceptance_probability(beta, gamma, image, random_pixel_position):
    # gamma = 0.5 * log((1 - pi) / pi)
    neighbors_energy = clique_energy(image, random_pixel_position)
    i, j = random_pixel_position
    current_pixel_value = image[i][j]
    # There's something wrong here TODO: change posterior calculation
    posterior = np.exp(
        -2 * gamma * current_pixel_value * current_pixel_value - 2 * beta * current_pixel_value * neighbors_energy)
    return posterior


# Reduces an image to 1-channel image
def reduce_channels_for_sampler(image):
    w = image.shape[0]
    h = image.shape[1]
    new_image = np.ndarray(shape=(w, h))  # create a new array for pixel values
    for i in range(w):  # rows
        for j in range(h):  # columns
            new_image[i][j] = image[i, j][:1]
    return new_image
    # if not all_elements_equal(image[i][j]):
    # print(image[i][j])


def restore_channels(image, n_channel):
    w = image.shape[0]
    h = image.shape[1]
    new_image = np.ndarray(shape=(w, h), dtype='int8')
    for i in range(w):  # rows
        for j in range(h):  # columns
            pixel = image[i, j] * 3
            pixels = image[i, j] * 3 #np.repeat(pixel, n_channel)
            new_image[i, j] = pixels #[pixel for n in range(n_channel)]
    return new_image


def within_boundaries(x, dim):  # check if pixel is within boundaries of a matrix
    width, height = dim[0], dim[1]
    return 0 <= x[0] < width and 0 <= x[1] < height


def choose_random_neighbor(neighbors):
    random_neighbor = random.choice(neighbors)
    return random_neighbor


# If diagonal_neighbors = True, the clique contains 8 sites
def get_all_neighbors(position, image_dimensions,
                      diagonal_neighbors=True):  # let's try passing tuple with pixel's position instead of one number
    i, j = position  # unpacking the tuple
    neighbors = []
    left_neighbor = (i, j - 1)
    right_neighbor = (i, j + 1)
    bottom_neighbor = (i + 1, j)
    top_neighbor = (i - 1, j)
    neighbors.extend(
        [left_neighbor, right_neighbor, bottom_neighbor, top_neighbor])
    # find diagonal elements
    if diagonal_neighbors:
        south_east_neighbor = (i + 1, j + 1)
        south_west_neighbor = (i + 1, j - 1)
        north_east_neighbor = (i - 1, j + 1)
        north_west_neighbor = (i - 1, j - 1)
        neighbors.extend(
            [south_east_neighbor, south_west_neighbor,
             north_east_neighbor, north_west_neighbor])
    neighbors = [x for x in neighbors if
                 within_boundaries(x,
                                   image_dimensions)]  # check if all coordinates are positive and each pixel lies within boundaries
    print("Neighbors for {0}:".format(position))
    print(neighbors)
    return neighbors


def run_metropolis_sampler(image):
    # arbitrary beta and gamma
    beta = 0.4
    gamma = 1.0
    pi = 0.1
    # print("Before:")
    # print(image.shape)
    image = reduce_channels_for_sampler(image)
    image = convert_image_to_ising_model(image)  # our prior belief of an image, X
    time = range(0, 1000)
    width, height = image.shape[0], image.shape[1]
    for t in time:
        i, j = random.randint(0, width - 1), random.randint(0, height - 1)
        flip_value = - image[i, j]
        pixel_position = (i, j)
        alpha = acceptance_probability(beta, gamma, image, pixel_position)
        if alpha >= 1:
            image[i][j] = flip_value
        else:
            random_number = random.random()
            if random_number <= alpha:
                image[i][j] = flip_value
    sampled_image = restore_channels(image, 3)
    sampled_image = convert_from_ising_to_image(sampled_image)
    cv2.imwrite('AfterSampling.png', sampled_image)


# Accepts a list
def all_elements_unique(items_list):
    items_set = set(items_list)
    if len(items_set) == len(items_list):
        return True
    return False


def all_elements_equal(items_list):
    items_set = set(items_list)
    if len(items_set) == 1:
        return True
    return False


def testing_images_structure(image):
    w = image.shape[0]
    h = image.shape[1]
    print(image[4, 4])
    # when iterating, remember that images may not be square
    T = 128
    # for i in range(w):  # rows
    # for j in range(h):  # columns
    # image[i, j] = 0 if (image[i, j] <= T).any() else 255
    # for k in range(len(image[i][j])):
    # if not all_elements_equal(image[i][j]):
    # print(image[i][j])


if __name__ == '__main__':
    image = cv2.imread('Images/cat_bw_noisy.png')
    print(image.shape)
    #run_metropolis_sampler(image)
    # testing_images_structure()
    # print(reduce_channels_for_sampler(image))
    # print(timeit.timeit(testing_images_structure, number=2))
    # 0print(len(image[6][10]))
