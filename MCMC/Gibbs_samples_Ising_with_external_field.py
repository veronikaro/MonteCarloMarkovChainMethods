from MCMC import metropolis_sampler
from MCMC import images_processing
from MCMC.services.auxiliary_methods import arithmetic_progression_series
import numpy as np
import random

NOISE_LEVEL = 0.05


# For the Ising model with an external field

def neighbors_energy(image, position):
    neighbors = metropolis_sampler.get_all_neighbors(position, image.shape)
    potentials = []
    for n in neighbors:
        i, j = n
        potentials.append(image[i, j])
    return np.sum(potentials)

def sum_of_all_spins(image):
    rows = range(image.shape[0])
    columns = range(image.shape[1])
    sum = 0
    for i in rows:
        for j in columns:
            sum += image[i, j]
    return sum


# Goal: to get a posterior distribution (the most probable image) by sampling
def run_gibbs_with_external_field(image):
    #image = images_processing.reduce_channels_for_sampler(image)
    image = images_processing.convert_image_to_ising_model(image)
    iterations = 2
    initial_beta = 0.3
    beta_difference = 0.1
    beta_range = arithmetic_progression_series(initial_beta, beta_difference, 10)
    rows = range(image.shape[0])
    columns = range(image.shape[1])
    beta = 1.3
    all_spins_energy = sum_of_all_spins(image)
    for t in range(iterations):
        for i in rows:
            for j in columns:
                site = (i, j)
                s = neighbors_energy(image, site) # a sum of markov blanket values
                #odds = np.exp(2*beta*s + 0.5*all_spins_energy)
                p = 1 / (1 + np.exp(-2*s))
                u = random.random()
                if u < p.any():
                    image[i, j] = 1
                else:
                    image[i, j] = -1
    sampled_image = images_processing.convert_from_ising_to_image(image)
    #sampled_image = images_processing.restore_channels(sampled_image, 3)  # restored image
    images_processing.save_image(
        'Denoised images/testing_no_channel_reduction/metropolis_noise_ising_chest_b={0}_noise={1}'.format(beta, NOISE_LEVEL),
        'jpeg', sampled_image)


# Test if all the pixel values are equal to either +1 or -1
def testing_conversion_to_ising(image):
    image = images_processing.convert_image_to_ising_model(image)
    h = image.shape[0]
    w = image.shape[1]
    for y in range(0, h):
        for x in range(0, w):
            if np.abs(image[y, x].any()) > 1:
                print(image[y, x])

if __name__ == '__main__':
    pass