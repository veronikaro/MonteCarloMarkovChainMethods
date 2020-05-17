from MCMC import metropolis_sampler
from MCMC import images_processing
from MCMC.services.auxiliary_methods import arithmetic_progression_series
from imghdr import what
import cv2
import numpy as np
import random
import sys

NOISE_LEVEL = 0.05


# This model of the posterior distribution doesn't use the knowledge about noise
def run_metropolis_without_noise(image_name, beta, iterations, neighbors_number):
    original_image = cv2.imread(image_name)
    original_image = images_processing.reduce_channels_for_sampler(original_image)
    original_image = images_processing.convert_image_to_ising_model(original_image)
    sampled_image = original_image
    rows = range(original_image.shape[0])
    columns = range(original_image.shape[1])
    for t in range(iterations):
        for i in rows:
            for j in columns:
                i = random.randint(0, rows - 1)
                j = random.randint(0, columns - 1)
                current_site = (i, j)
                flipped_value = - sampled_image[current_site]
                d = beta * metropolis_sampler.potentials(sampled_image, current_site, flipped_pixel_value=False,
                                                         neighbors_number=neighbors_number)
                # the conditional probability for the opposite to the random pixel's value
                d_flipped = beta * metropolis_sampler.potentials(sampled_image, current_site,
                                                                 flipped_pixel_value=True)
                posterior = np.exp(min(d_flipped - d, 0))
                u = random.random()
                if u < posterior:
                    sampled_image[current_site] = flipped_value
    sampled_image = images_processing.convert_from_ising_to_image(sampled_image)
    sampled_image = images_processing.restore_channels(sampled_image, 3)
    format = what(image_name)
    images_processing.save_image(
        'result_beta={0}_iter={1}_neighbors={2}'.format(beta, iterations, neighbors_number), format, sampled_image,
        directory='Results')
    print('success')


def run_metropolis_without_noise_for_beta_range(image_name, initial_beta, beta_step, iterations, neighbors_number):
    original_image = cv2.imread(image_name)
    original_image = images_processing.reduce_channels_for_sampler(original_image)
    original_image = images_processing.convert_image_to_ising_model(original_image)
    rows = range(original_image.shape[0])
    columns = range(original_image.shape[1])
    beta_range = arithmetic_progression_series(initial_beta, beta_step, 10)
    for beta in beta_range:
        sampled_image = original_image  # for different values of beta, reset the sampled image to the original one
        for t in range(iterations):
            for i in rows:
                for j in columns:
                    i = random.randint(0, rows - 1)
                    j = random.randint(0, columns - 1)
                    current_site = (i, j)
                    flipped_value = - sampled_image[current_site]
                    d = beta * metropolis_sampler.potentials(sampled_image, current_site, flipped_pixel_value=False,
                                                             neighbors_number=neighbors_number)
                    d_flipped = beta * metropolis_sampler.potentials(sampled_image, current_site,
                                                                     flipped_pixel_value=True)
                    posterior = np.exp(min(d_flipped - d, 0))
                    u = random.random()
                    if u < posterior:
                        sampled_image[current_site] = flipped_value
        sampled_image = images_processing.convert_from_ising_to_image(sampled_image)
        sampled_image = images_processing.restore_channels(sampled_image, 3)
        format = what(image_name)
        images_processing.save_image(
            'result_beta={0}_iter={1}_neighbors={2}'.format(beta, iterations, neighbors_number), format, sampled_image,
            directory='Results')


if __name__ == '__main__':
    # noised = cv2.imread('brain2_bw_noised.jpg')
    arguments = sys.argv
    image_name, beta, iterations, neighbors_number = arguments[1:]
    run_metropolis_without_noise(image_name, float(beta), int(iterations), int(neighbors_number))
