from MCMC import metropolis_sampler
from MCMC import noiser
from MCMC import images_processing
from MCMC.auxiliary_methods import arithmetic_progression_series

import cv2
import numpy as np
import random
import datetime

# Works nice

# Constants
NOISE_LEVEL = 0.1  # a prior knowledge about noise level (or expected noise)
ITERATIONS_NUMBER = 5  # it takes approximately 04:13 minutes to iterate 5 times over an image of size 1457724 ~ 1.5mln pixels
SIGMA = 1


def noise(p):
    return 0.5 * np.log((1 - p) / p)


# additive Gaussian noise
# TODO: test the model with Gaussian noise
def gaussian_noise(sigma, yi, xi):  # h(x, y) function
    return - 0.5/(sigma**2) * (yi - xi)**2


def run_metropolis_with_noise(image):
    image = metropolis_sampler.reduce_channels_for_sampler(image)
    image = metropolis_sampler.convert_image_to_ising_model(image)
    iterations = ITERATIONS_NUMBER
    initial_beta = 0.3
    beta_difference = 0.1
    # beta_range = arithmetic_progression_series(initial_beta, beta_difference, 10)
    beta_range = [1.3]
    rows = range(image.shape[0])
    columns = range(image.shape[1])
    for beta in beta_range:
        for t in range(iterations):
            for i in rows:
                for j in columns:
                    site = (i, j)
                    flipped_value = - image[site]
                    d = beta * metropolis_sampler.potentials(image, site) + gaussian_noise(SIGMA)
                    d_stroke = beta * metropolis_sampler.potentials(image, site,
                                                                    flipped_pixel_value=True) + gaussian_noise(SIGMA)
                    # d = beta * metropolis_sampler.potentials(image, site) + noise(NOISE_LEVEL)
                    # d_stroke = beta * metropolis_sampler.potentials(image, site, flipped_pixel_value=True) + noise(
                    # NOISE_LEVEL)
                    posterior = np.exp(min(d_stroke - d, 0))
                    u = random.random()
                    if u < posterior:
                        image[site] = flipped_value
        sampled_image = metropolis_sampler.convert_from_ising_to_image(image)
        sampled_image = metropolis_sampler.restore_channels(sampled_image, 3)  # restored image
        images_processing.save_image('metropolis_noise_ising_beta={0}_iter={1}'.format(beta, iterations), 'jpeg',
                                     sampled_image,
                                     'Denoised/metropolis_sampler_noise_ising_model/noise={0}'.format(NOISE_LEVEL))


if __name__ == '__main__':
    # start = datetime.datetime.now()
    image = cv2.imread('noised_chest10%.jpeg')
    print(image.shape[0] * image.shape[1])
    # run_metropolis_with_noise(image)
    # print('time since start of the sampling process:')
    # print(datetime.datetime.now() - start)
    # image = noising.sp_noise(image, 0.05)
    # noising.save_image('noised_chest', 'jpeg', image)
    # images_processing.convert_to_bw_and_greyscale(image, 'chest', 'jpeg')


## Note: the potentials() function works as expected
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
