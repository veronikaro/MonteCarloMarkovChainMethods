from MCMC import metropolis_sampler
from MCMC import images_processing
from MCMC.services.auxiliary_methods import arithmetic_progression_series
import cv2
import numpy as np
import random
import imghdr
# Constants
NOISE_LEVEL = 0.05  # a prior knowledge about noise level (or expected noise)
ITERATIONS_NUMBER = 5  # it takes approximately 04:13 minutes to iterate 5 times over an image of size 1457724 ~ 1.5mln pixels
SIGMA = 1


def noise(p, sampled_pixel_value, original_pixel_value):
    return metropolis_sampler.indicator_func(sampled_pixel_value, original_pixel_value) * np.log((1 - p) / p)


# TODO: test the model with Gaussian noise
def gaussian_noise(sigma):  # h(x, y) function
    """
    :return: additive Gaussian noise
    """
    return - 0.5 / (sigma ** 2) * (0.5) ** 2


def run_metropolis_with_noise(image):
    """
    Runs the Metropolis sampler with a strategy to choose the next pixel to update non-randomly.
    :param image: an image of size m x n
    :return: saves the result of denoising to the given folder
    """
    image = images_processing.reduce_channels_for_sampler(image)
    image = images_processing.convert_image_to_ising_model(image)
    iterations = ITERATIONS_NUMBER
    initial_beta = 0.3
    beta_difference = 0.1
    # beta_range = arithmetic_progression_series(initial_beta, beta_difference, 12)
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
        sampled_image = images_processing.convert_from_ising_to_image(image)
        sampled_image = images_processing.restore_channels(sampled_image, 3)  # restored image
        images_processing.save_image(
            'metropolis_noise_ising_beta={0}_iter={1}'.format(beta, iterations), 'jpg',
            sampled_image,
            'Denoised images/metropolis_sampler_noise_ising_model/noise={0}/brain_tissue_im/not_random'.format(
                NOISE_LEVEL))


def run_random_metropolis_with_noise(image):
    image = metropolis_sampler.reduce_channels_for_sampler(image)
    original_image = metropolis_sampler.convert_image_to_ising_model(image)
    sampled_image = original_image
    iterations = image.shape[0] * image.shape[1]  # the overall number of pixels
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

    images_processing.save_image('updated_noise_model_4neighbors_beta={0}'.format(beta), 'jpg', sampled_image,
                                 '')

# make it possible to run the script from the command line. the possible requirement is to run this script from the directory where the target image is located
def denoising_pipeline(image_name, beta, iterations, noise_probability, neighbors_number):
    # read the image
    original_image = cv2.imread(image_name)
    # reduce channels
    original_image = images_processing.reduce_channels_for_sampler(original_image)
    # convert to Ising
    original_image = images_processing.convert_image_to_ising_model(original_image)
    # create a copy of the image to which the changes will be applied
    sampled_image = original_image
    # accept beta as an argument
    # accept iterations number as an argument
    # accept noise probability as an argument
    # accept the neighbourhood structure's size
    # get image dimensions
    rows = original_image.shape[0]  # height
    columns = original_image.shape[1]  # width

    for t in range(iterations):
        # generate a random pixel position
        i = random.randint(0, rows - 1)
        j = random.randint(0, columns - 1)
        current_site = (i, j)
        # find the opposite value of the current pixel
        flipped_value = - sampled_image[current_site]
        # the conditional probability for the random pixel's value
        d = beta * metropolis_sampler.potentials(sampled_image, current_site, flipped_pixel_value=False,
                                                 neighbors_number=neighbors_number) + noise(noise_probability,
                                                                                            sampled_image[current_site],
                                                                                            original_image[
                                                                                                current_site])
        # the conditional probability for the opposite to the random pixel's value
        d_flipped = beta * metropolis_sampler.potentials(sampled_image, current_site, flipped_pixel_value=True) + noise(
            noise_probability,
            - sampled_image[current_site], original_image[current_site])
        posterior = np.exp(min(d_flipped - d, 0))
        u = random.random()
        if u < posterior:
            sampled_image[current_site] = flipped_value
    # convert back from Ising
    sampled_image = images_processing.convert_from_ising_to_image(sampled_image)
    # restore channels
    sampled_image = images_processing.restore_channels(sampled_image, 3)  # restored image
    # create a separate folder to save the result with parameters specified
    format = imghdr.what(image_name)
    print('success')
    images_processing.save_image('result_beta={}_noise_p={}_iter={}_neighbors={}', format, sampled_image, directory='Results')


if __name__ == '__main__':
    #start = datetime.datetime.now()
    original_image = cv2.imread('Noisy images/noised_10.0%_grumpy_cat.jpg')
    denoising_pipeline('Noisy images/noised_10.0%_grumpy_cat.jpg', 0.8, 1000, 0.1, 8)