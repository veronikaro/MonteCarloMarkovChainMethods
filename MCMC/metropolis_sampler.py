import numpy as np
import timeit
import random
import cython
from MCMC import images_processing
from math import exp, log
import cv2

# Not used
def clique_energy(image, pixel_position):
    """Calculate the local energy of pixels in the neighborhood of a given pixel."""
    neighbors = get_all_neighbors(pixel_position, image.shape)
    pixel_intensities = []
    for n in neighbors:
        pixel_intensities.append(image[n])
    return sum(pixel_intensities)


def indicator_func(x, y):
    if x == y:
        return 1
    else:
        return 0


# Return the potential energy of the system -> energy of the neighborhood
def potentials(image, position, flipped_pixel_value=False, neighbors_number=8):
    if flipped_pixel_value:
        current_pixel_value = -image[position]
    else:
        current_pixel_value = image[position]
    if neighbors_number == 8:
        neighbors = get_all_neighbors(position, image.shape)
    else:
        neighbors = get_all_neighbors(position, image.shape, diagonal_neighbors=False)
    energy = 0
    for n in neighbors:
        energy = energy + indicator_func(image[n], current_pixel_value)
    return energy


# Initial trying, not used
def acceptance_probability(beta, pi, init_image, current_image, random_pixel_position):
    """
    Calculate an acceptance probability of flipping a given pixel.

    :param beta: the strength of coupling (interaction) between pixels
    :param pi: ?
    :param image: a monochrome image with pixel intensities converted to -1 and +1; our prior belief of an image, X
    :param random_pixel_position: coordinates of a given pixel to be flipped or not
    :return: a floating point number - posterior probability
    """
    # TODO: J is a coupling strength between pixels. Its sign tells if spins prefer to align or to anti-align
    # TODO: another parameter h - external field
    gamma = 0.5 * log((1 - pi) / pi)  # external factor. also called J - a coupling strength
    neighbors_energy = clique_energy(current_image, random_pixel_position)
    i, j = random_pixel_position
    init_pixel_value = init_image[i][j]
    current_pixel_value = current_image[i][j]
    posterior = -2 * gamma * init_pixel_value * current_pixel_value - 2 * beta * current_pixel_value * neighbors_energy  # posterior function
    return posterior


def within_boundaries(x, dim):
    """Check if a pixel lies within boundaries of a matrix."""
    width, height = dim[0], dim[1]
    return 0 <= x[0] < width and 0 <= x[1] < height


def choose_random_neighbor(neighbors):
    """Choose a random neighbor from the given ones."""
    random_neighbor = random.choice(neighbors)
    return random_neighbor


# Get Markov blanket for a given pixel
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


# TODO: re-think logic of this version of the algorithm. Not used now
# Works with images converted to the binary format (with 0 and 255 as pixel values)
def run_metropolis_sampler(image):
    """Run the Metropolis sampler for the given noised image."""
    # arbitrary beta and pi TODO: How beta, pi and gamma can be interpreted?
    beta = 0.8
    # gamma = 1.0
    pi = 0.05
    # print("Before:")
    # print(image.shape)
    image = images_processing.reduce_channels_for_sampler(image)  # convert a 3-channel image to 1-channel one
    image = images_processing.convert_image_to_ising_model(image)  # initial image
    init_image = image
    current_image = init_image
    width, height = image.shape[0], image.shape[1]  # dimensions
    iterations = range(width * height)
    for t in iterations:
        i, j = random.randint(0, width - 1), random.randint(0, height - 1)
        flipped_value = - image[i, j]
        pixel_position = (i, j)
        alpha = acceptance_probability(beta, pi, init_image, current_image, pixel_position)
        random_number = random.random()
        if np.log(random_number) < alpha.any():
            current_image[i][j] = flipped_value
    sampled_image = images_processing.convert_from_ising_to_image(current_image)
    sampled_image = images_processing.restore_channels(sampled_image, 3)
    print(sampled_image.shape)
    cv2.imwrite('testingiter={}.jpg'.format(len(iterations)), sampled_image)  # should be a separate function


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
    image = cv2.imread('brain4_noise5%.jpg')
    run_metropolis_sampler(image)
