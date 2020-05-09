from MCMC import metropolis_sampler
from MCMC import noiser
from MCMC import images_processing
from MCMC.auxiliary_methods import arithmetic_progression_series

import cv2
import numpy as np
import random

NOISE_LEVEL = 0.05


# calculate the number of black or white neighbors

def energy_gibbs(image, position, color='black'):
    neighbors = metropolis_sampler.get_all_neighbors(position, image.shape)
    required_neighbors_number = 0

    for n in neighbors:
        if color == 'black' and image[n] == -1:
            required_neighbors_number += 1
        elif color == 'white' and image[n] == 1:
            required_neighbors_number += 1
    return required_neighbors_number


def run_gibbs_without_noise(image):
    image = metropolis_sampler.reduce_channels_for_sampler(image)
    image = metropolis_sampler.convert_image_to_ising_model(image)
    iterations = 100
    # beta = 1.3
    initial_beta = 0.3
    beta_difference = 0.1
    beta_range = arithmetic_progression_series(initial_beta, beta_difference, 10)
    beta_range = [1.3]
    # noise_prob = 0.05  # this parameter is taken from the knowledge of noise level in the image
    rows = range(image.shape[0])
    columns = range(image.shape[1])
    for beta in beta_range:
        for t in range(iterations):
            for i in rows:
                for j in columns:
                    site = (i, j)
                    number_of_black_neighbors = energy_gibbs(image, site, 'black')
                    number_of_white_neighbors = energy_gibbs(image, site, 'white')
                    Z_normalizing_constant = np.exp(number_of_black_neighbors) + np.exp(number_of_white_neighbors)
                    posterior = np.exp(beta*number_of_black_neighbors)/Z_normalizing_constant
                    u = random.random()
                    if u < posterior:
                        image[site] = 1
                    else:
                        image[site] = -1
        sampled_image = metropolis_sampler.convert_from_ising_to_image(image)
        sampled_image = metropolis_sampler.restore_channels(sampled_image, 3)  # restored image
        images_processing.save_image(
            'Denoised/gibbs_sampler_no_noise/metropolis_noise_ising_chest_b={0}_noise={1}'.format(beta, NOISE_LEVEL),
            'jpeg', sampled_image)


if __name__ == '__main__':
    image = cv2.imread('noised_chest5%.jpeg')
    run_gibbs_without_noise(image)
    # image = noising.sp_noise(image, 0.05)
    # noising.save_image('noised_chest', 'jpeg', image)
    # images_processing.convert_to_bw_and_greyscale(image, 'chest', 'jpeg')


## Note: potentials() function works as expected
def testing_potentials_calculation(image):
    image = metropolis_sampler.reduce_channels_for_sampler(image)
    image = metropolis_sampler.convert_image_to_ising_model(image)
    energy = metropolis_sampler.potentials(image, (5, 5))
    print("current pixel value")
    print(image[(5, 5)])
    neighbors = metropolis_sampler.get_all_neighbors((5, 5), image.shape)
    print("values of neighbors")
    for coord in neighbors:
        print(image[coord])
    print("energy")
    print(energy)
