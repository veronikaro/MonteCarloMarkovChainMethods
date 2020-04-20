import numpy as np
import timeit
import random
import cython
from math import exp, log
from MCMC.metropolis_sampler import clique_energy, restore_channels, reduce_channels_for_sampler, \
    convert_image_to_ising_model, convert_from_ising_to_image
import cv2


# TODO: check if energy is calculated the same way as in Metropolis case



def acceptance_probability(beta, pi, image, random_pixel_position):
    """Calculate an acceptance probability of flipping a given pixel.

        Keyword arguments:
        beta -- ?
        pi -- ?
        image -- a monochrome image with pixel intensities converted to -1 and +1; our prior belief of an image, X
        random_pixel_position -- pixel to be flipped or not
        """
    gamma = 0.5 * log((1 - pi) / pi)  # external_factor [
    neighbors_energy = clique_energy(image, random_pixel_position)
    i, j = random_pixel_position
    current_pixel_value = image[i][j]
    # There's something wrong here TODO: change posterior calculation
    posterior = -2 * gamma * current_pixel_value * current_pixel_value - 2 * beta * current_pixel_value * neighbors_energy  # posterior function
    return posterior



if __name__ == '__main__':

    print("hello, I'm main")
