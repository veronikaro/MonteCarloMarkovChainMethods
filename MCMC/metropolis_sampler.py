import numpy as np
import timeit
import random
import cython
from math import exp, log
from MCMC.images_processing import *
import cv2


# TODO: describe all parameters used in the algorithm

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


def clique_energy(image, pixel_position):
    """Calculate the local energy of pixels in the neighborhood of a given pixel."""
    neighbors = get_all_neighbors(pixel_position, image.shape)
    pixel_intensities = []
    for n in neighbors:
        pixel_intensities.append(image[n])
    return sum(pixel_intensities)


def acceptance_probability(beta, pi, image, random_pixel_position):
    """
    Calculate an acceptance probability of flipping a given pixel.

    :param beta: ?
    :param pi: ?
    :param image: a monochrome image with pixel intensities converted to -1 and +1; our prior belief of an image, X
    :param random_pixel_position: coordinates of a given pixel to be flipped or not
    :return: a floating point number - posterior probability
    """
    gamma = 0.5 * log((1 - pi) / pi)  # external_factor
    neighbors_energy = clique_energy(image, random_pixel_position)
    i, j = random_pixel_position
    current_pixel_value = image[i][j]
    # TODO: double-check posterior calculation
    posterior = -2 * gamma * current_pixel_value * current_pixel_value - 2 * beta * current_pixel_value * neighbors_energy  # posterior function
    return posterior


# TODO: think how to improve for better performance when iterating over pixels.
def reduce_channels_for_sampler(image):
    """Reduce a 3-channel image to 1-channel image."""
    w = image.shape[0]
    h = image.shape[1]
    new_image = np.ndarray(shape=(w, h))  # create a new array for pixel values
    for i in range(w):  # rows
        for j in range(h):  # columns
            new_image[i][j] = image[i, j][:1]
    return new_image


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


def within_boundaries(x, dim):
    """Check if a pixel lies within boundaries of a matrix."""
    width, height = dim[0], dim[1]
    return 0 <= x[0] < width and 0 <= x[1] < height


def choose_random_neighbor(neighbors):
    """Choose a random neighbor from the given ones."""
    random_neighbor = random.choice(neighbors)
    return random_neighbor


# If diagonal_neighbors = True, the clique contains 8 sites
def get_all_neighbors(position, image_dimensions,
                      diagonal_neighbors=True):  # let's try passing tuple with pixel's position instead of one number
    """
    Find the given pixel's neighbors

    :param position: a tuple with two pixel's coordinates
    :param image_dimensions: a tuple with image's width and height
    :param diagonal_neighbors: boolean to decide whether to find diagonals neighbors of a given pixel. If True, the clique contains 8 sites. Default: True
    :return: a list of tuples with coordinates of the given pixel's neighbors
    """
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
    # print("Neighbors for {0}:".format(position)) # for debugging: checking if neighbors are defined correctly
    # print(neighbors)
    return neighbors


def run_metropolis_sampler(image):
    """Run the Metropolis sampler for the given noised image."""
    # arbitrary beta and pi TODO: How beta, pi and gamma can be interpreted?
    beta = 0.8
    # gamma = 1.0
    pi = 0.15
    # print("Before:")
    # print(image.shape)
    image = reduce_channels_for_sampler(image)  # convert a 3-channel image to 1-channel one
    image = convert_image_to_ising_model(image)
    time = range(0, 2000000)
    width, height = image.shape[0], image.shape[1]  # dimensions
    for t in time:
        i, j = random.randint(0, width - 1), random.randint(0, height - 1)
        flipped_value = - image[i, j]
        pixel_position = (i, j)
        alpha = acceptance_probability(beta, pi, image, pixel_position)
        random_number = random.random()
        if np.log(random_number) < alpha:
            image[i][j] = flipped_value
    sampled_image = convert_from_ising_to_image(image)
    sampled_image = restore_channels(sampled_image, 3)
    print(sampled_image.shape)
    cv2.imwrite('DenoisedIm_Iter={}.png'.format(len(time)), sampled_image)


def all_elements_unique(items_list):
    """Check if all elements on the list are unique."""
    items_set = set(items_list)
    if len(items_set) == len(items_list):
        return True
    return False


def all_elements_equal(items_list):
    """Check if all elements on the list are equal."""
    items_set = set(items_list)
    if len(items_set) == 1:
        return True
    return False


if __name__ == '__main__':
    image = cv2.imread('Images/cat_bw_noisy.png')
    run_metropolis_sampler(image)
    # print(image.shape)
    # t = timeit.Timer('run_metropolis_sampler(image)', globals=locals())
    # print(t.timeit(number=3))
    # t = timeit.Timer(run_metropolis_sampler(image))
    # timeit.timeit('run_metropolis_sampler(image), 'from __main__ import run_metropolis_sampler, image', number=3)
    # print(alg_time)
    # testing_images_structure()
    # print(reduce_channels_for_sampler(image))
    # print(timeit.timeit(testing_images_structure, number=2))
    # 0print(len(image[6][10]))

'''
def restore_channels(image, n_channel):
    w = image.shape[0]
    h = image.shape[1]
    new_image = np.ndarray(shape=(w, h), dtype='int8')
    for i in range(w):  # rows
        for j in range(h):  # columns
            pixels = image[i, j] * n_channel #np.repeat(pixel, n_channel)
            new_image[i, j] = pixels #[pixel for n in range(n_channel)]
    return new_image
    
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

'''
