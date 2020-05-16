from MCMC import metropolis_sampler
from MCMC import images_processing
from MCMC.services.auxiliary_methods import arithmetic_progression_series
from math import exp, log
import cv2
import numpy as np
import random

# Works nice

# Constants
NOISE_LEVEL = 0.05  # a prior knowledge about noise level (or expected noise)
ITERATIONS_NUMBER = 5  # it takes approximately 04:13 minutes to iterate 5 times over an image of size 1457724 ~ 1.5mln pixels
SIGMA = 1


def noise(p, sampled_pixel_value, original_pixel_value):
    return metropolis_sampler.indicator_func(sampled_pixel_value, original_pixel_value) * np.log((1 - p) / p)


# additive Gaussian noise
# TODO: test the model with Gaussian noise
def gaussian_noise(sigma):  # h(x, y) function
    return - 0.5 / (sigma ** 2) * (0.5) ** 2


def run_metropolis_with_noise(image):
    image = metropolis_sampler.reduce_channels_for_sampler(image)
    image = metropolis_sampler.convert_image_to_ising_model(image)
    iterations = ITERATIONS_NUMBER
    initial_beta = 0.3
    beta_difference = 0.1
    beta_range = arithmetic_progression_series(initial_beta, beta_difference, 12)
    beta_range = [0.8]
    rows = range(image.shape[0])
    columns = range(image.shape[1])
    for beta in beta_range:
        for t in range(iterations):
            for i in rows:
                for j in columns:
                    site = (i, j)
                    flipped_value = - image[site]
                    d = beta * metropolis_sampler.potentials(image, site) + noise(NOISE_LEVEL)
                    d_stroke = beta * metropolis_sampler.potentials(image, site, flipped_pixel_value=True) + noise(
                        NOISE_LEVEL)
                    posterior = np.exp(min(d_stroke - d, 0))
                    u = random.random()
                    if u < posterior:
                        image[site] = flipped_value
        sampled_image = metropolis_sampler.convert_from_ising_to_image(image)
        sampled_image = metropolis_sampler.restore_channels(sampled_image, 3)  # restored image
        images_processing.save_image(
            'metropolis_noise_ising_beta={0}_iter={1}'.format(beta, iterations), 'jpg',
            sampled_image,
            'Denoised images/metropolis_sampler_noise_ising_model/noise={0}/brain_tissue_im/not_random'.format(
                NOISE_LEVEL))


def run_random_metropolis_with_noise(image):
    image = metropolis_sampler.reduce_channels_for_sampler(image)
    original_image = metropolis_sampler.convert_image_to_ising_model(image)
    sampled_image = original_image
    iterations = 10 * image.shape[0] * image.shape[1]  # the overall number of pixels
    initial_beta = 0.3
    beta_difference = 0.1  # delta
    # beta_range = arithmetic_progression_series(initial_beta, beta_difference, 10)
    beta = 0.9
    rows = image.shape[0]
    columns = image.shape[1]
    for t in range(iterations):
        i = random.randint(0, rows - 1)
        j = random.randint(0, columns - 1)
        site = (i, j)
        flipped_value = - sampled_image[site]
        d = beta * metropolis_sampler.potentials(sampled_image, site) + noise(NOISE_LEVEL, sampled_image[site],
                                                                              original_image[site])
        d_stroke = beta * metropolis_sampler.potentials(sampled_image, site, flipped_pixel_value=True) + noise(
            NOISE_LEVEL,
            - sampled_image[site], original_image[site])
        posterior = np.exp(min(d_stroke - d, 0))
        u = random.random()
        if u < posterior:
            sampled_image[site] = flipped_value
    sampled_image = metropolis_sampler.convert_from_ising_to_image(sampled_image)
    sampled_image = metropolis_sampler.restore_channels(sampled_image, 3)  # restored image
    # save to the current directory (testing)

    images_processing.save_image('10x_iters_updated_noise_model_4neighbors_beta={0}'.format(beta), 'jpg', sampled_image,
                                 '')


# make it possible to run the script from the command line. the possible requirement is to run this script from the directory where the target image is located
def denoising_pipeline():
    # read image
    # reduce channels
    # convert to Ising
    # accept beta as an argument
    # accept iterations number as an argument
    # accept noise probability as an argument

    # create a separate folder to save the result with parameters specified
    pass


def run_random_metropolis_with_noise_experiment(image):
    image = metropolis_sampler.reduce_channels_for_sampler(image)
    image = metropolis_sampler.convert_image_to_ising_model(image)
    init_im = image
    iterations = image.shape[0] * image.shape[1]  # the overall number of pixels
    initial_beta = 0.3
    beta_difference = 0.1  # delta
    beta_range = arithmetic_progression_series(initial_beta, beta_difference, 20)
    # beta = 0.8
    for beta in beta_range:
        for t in range(iterations):
            i = random.randint(0, image.shape[0] - 1)
            j = random.randint(0, image.shape[1] - 1)
            site = (i, j)
            flipped_value = - image[site]
            alpha = acceptance_probability(beta, NOISE_LEVEL, init_im, image, site)
            u = random.random()
            if np.log(u) < alpha.any():
                image[site] = flipped_value
        sampled_image = metropolis_sampler.convert_from_ising_to_image(image)
        sampled_image = metropolis_sampler.restore_channels(sampled_image, 3)  # restored image
        images_processing.save_image('experiment_beta={0}'.format(beta), 'jpg', sampled_image, '')


def acceptance_probability(beta, pi, init_image, current_image, random_pixel_position):
    """
    Calculate an acceptance probability of flipping a given pixel.

    :param beta: the strength of coupling (interaction) between pixels
    :param pi: ?
    :param image: a monochrome image with pixel intensities converted to -1 and +1; our prior belief of an image, X
    :param random_pixel_position: coordinates of a given pixel to be flipped or not
    :return: a floating point number - posterior probability
    """
    gamma = 0.5 * log((1 - pi) / pi)  # external factor. also called J - a coupling strength
    neighbors_energy = metropolis_sampler.potentials(current_image, random_pixel_position)
    i, j = random_pixel_position
    init_pixel_value = init_image[i][j]
    current_pixel_value = current_image[i][j]
    posterior = -2 * gamma * init_pixel_value * current_pixel_value - 2 * beta * current_pixel_value * neighbors_energy  # posterior function
    return posterior


# d = beta * metropolis_sampler.potentials(image, site) + gaussian_noise(SIGMA)
# d_stroke = beta * metropolis_sampler.potentials(image, site,
# flipped_pixel_value=True) + gaussian_noise(SIGMA)


if __name__ == '__main__':
    # start = datetime.datetime.now()
    noised = cv2.imread('brain4_noise5%.jpg')
    run_random_metropolis_with_noise(noised)
