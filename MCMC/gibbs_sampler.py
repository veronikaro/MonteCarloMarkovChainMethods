import numpy as np
import timeit
import random
import cython
from math import exp, log
from MCMC.metropolis_sampler import clique_energy, restore_channels, reduce_channels_for_sampler, \
    convert_image_to_ising_model, convert_from_ising_to_image
import cv2


def probability(beta, external_strength, image, random_pixel_position):
    """
    Calculate an acceptance probability of flipping a given pixel.

    :param beta: the strength of coupling (interaction) between pixels
    :param external_strength: ?
    :param image: a monochrome image with pixel intensities converted to -1 and +1; our prior belief of an image, X
    :param random_pixel_position: coordinates of a given pixel to be flipped or not
    :return: a floating point number - posterior probability
    """
    neighbors_energy = clique_energy(image, random_pixel_position)
    i, j = random_pixel_position
    current_pixel_value = image[i][j]
    E = 2 * beta * neighbors_energy
    E = E + 2 * external_strength * current_pixel_value
    prob = 1. / (1 + np.exp(-E))
    return prob


def run_gibbs_sampler(image):
    """Run the Gibbs sampler for the given noised image."""
    beta = 0.8
    external_strength = 2
    image = reduce_channels_for_sampler(image)  # convert a 3-channel image to 1-channel one
    image = convert_image_to_ising_model(image)  # initial image
    # initial image
    sampled_image = image
    temperature = range(0, 100)
    width, height = image.shape[0], image.shape[1]  # dimensions
    for t in temperature:
        for i in range(width):  # rows
            for j in range(height):  # columns
                prob = probability(beta, external_strength, image, (i, j))
                flipped_value = - image[i, j]
                random_number = random.random()
                if np.log(random_number) < prob.any():
                    sampled_image[i][j] = flipped_value
    sampled_image = convert_from_ising_to_image(sampled_image)
    sampled_image = restore_channels(sampled_image, 3)
    print(sampled_image.shape)
    cv2.imwrite('gibbs_sampler_im_iter={}.png'.format(len(temperature)), sampled_image)


if __name__ == '__main__':
    im = cv2.imread('Images/cat_bw_noisy.png')
    run_gibbs_sampler(im)
