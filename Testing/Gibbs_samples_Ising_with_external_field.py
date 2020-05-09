from MCMC import metropolis_sampler
from MCMC import noiser
from MCMC import images_processing
from MCMC.auxiliary_methods import arithmetic_progression_series

import cv2
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
    #image = metropolis_sampler.reduce_channels_for_sampler(image)
    image = metropolis_sampler.convert_image_to_ising_model(image)
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
    sampled_image = metropolis_sampler.convert_from_ising_to_image(image)
    #sampled_image = metropolis_sampler.restore_channels(sampled_image, 3)  # restored image
    images_processing.save_image(
        'Denoised/testing_no_channel_reduction/metropolis_noise_ising_chest_b={0}_noise={1}'.format(beta, NOISE_LEVEL),
        'jpeg', sampled_image)


## Note: potentials() function works as expected
def testing_potentials_calculation(image):
    image = metropolis_sampler.reduce_channels_for_sampler(image)
    image = metropolis_sampler.convert_image_to_ising_model(image)
    # energy = metropolis_sampler.potentials(image, (5, 5))
    energy = neighbors_energy(image, (5, 5))
    print("current pixel value")
    print(image[(5, 5)])
    neighbors = metropolis_sampler.get_all_neighbors((5, 5), image.shape)
    print("values of neighbors")
    for coord in neighbors:
        print(image[coord])
    print("energy")
    print(energy)


# Test if all the pixel values are equal to either +1 or -1
def testing_conversion_to_ising(image):
    image = metropolis_sampler.convert_image_to_ising_model(image)
    h = image.shape[0]
    w = image.shape[1]
    for y in range(0, h):
        for x in range(0, w):
            if np.abs(image[y, x].any()) > 1:
                print(image[y, x])


def testing_channels_reduction(image):
    image = metropolis_sampler.reduce_channels_for_sampler(image)
    h = image.shape[0]
    w = image.shape[1]
    for y in range(0, h):
        for x in range(0, w):
            if not isinstance(image[y, x],
                              np.float64):  # we converted the array with 3 channels to an array with 1 channel. therefore, each element must be one number (of type float64 - default type of ndarray)
                print(image[y, x])
            else:
                if (image[
                    y, x] != 255 and image[
                    y, x] != 0):  # check also if all the values that are left after conversion to b&w image, are 0 or 255 (binary)
                    print(image[y, x])


if __name__ == '__main__':
    # image = cv2.imread('noised_chest5%.jpeg')
    # image = cv2.imread('chest_bw.jpeg')
    # image = noiser.sp_noise(image, 0.05)
    # print('before:')
    # print(image.shape)
    # image = metropolis_sampler.reduce_channels_for_sampler(image)
    # print('after: ')
    # print(image.shape)
    image = cv2.imread('noised_chest5%.jpeg')
    run_gibbs_with_external_field(image)
    '''
    image1 = cv2.imread('chest.jpeg')
    print('before:')
    print(image1.shape)
    image1 = images_processing.convert_to_bw_and_return(image1)
    print(image1)

    print('after:')
    print(image1.shape)
    testing_channels_reduction(image1)'''
    # testing_channels_reduction(image)
    # print(image.item(5, 5, 1))
    # print(image[5, 5])
    # print('image size:')
    # print(image.shape)
    # image = images_processing.convert_to_bw_and_greyscale('chest.jpeg')
    # print(image)
    # testing_potentials_calculation(image)
    # run_gibbs_without_noise(image)
    # image = noising.sp_noise(image, 0.05)
    # noising.save_image('noised_chest', 'jpeg', image)
    # images_processing.convert_to_bw_and_greyscale(image, 'chest', 'jpeg')
