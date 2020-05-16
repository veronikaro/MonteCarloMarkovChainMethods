from MCMC import metropolis_sampler
from MCMC import images_processing
import cv2
import sys
import numpy as np
import random
from imghdr import what


def noise(p, sampled_pixel_value, original_pixel_value):
    return metropolis_sampler.indicator_func(sampled_pixel_value, original_pixel_value) * np.log((1 - p) / p)


# Run the script from the command line and from the directory where the target image is located (!!!)
# Pass the arguments in the given order
# Example of execution:
# > python denoiser.py brain4_noise5%.jpg 0.8 100000 0.05 8
# Note that Python 3.x must be installed on your machine
def denoising_pipeline(image_name, beta, iterations, noise_probability, neighbors_number):
    """
    Performs denoising of a given image using the Metropolis sampler.
    :param image_name: a name of the image located in the current directory (from which the script is called)
    :param beta: an essential parameter of the sampler, an inverse to the temperature (1/T)
    :param iterations: the number of iterations
    :param noise_probability: the assumed probability of the flip noise in the image
    :param neighbors_number: the size of the neighborhood structure. it's used when calculating the energy of the neighborhood. the default is 8. can be switched to 4
    :return: saves a result of denoising to the folder 'Results' (it's created if doesn't exist)
    """
    # read the image
    original_image = cv2.imread(image_name)
    # reduce channels
    original_image = metropolis_sampler.reduce_channels_for_sampler(original_image)
    # convert to Ising
    original_image = metropolis_sampler.convert_image_to_ising_model(original_image)
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
    sampled_image = metropolis_sampler.convert_from_ising_to_image(sampled_image)
    # restore channels
    sampled_image = metropolis_sampler.restore_channels(sampled_image, 3)  # restored image
    # create a separate folder to save the result with parameters specified
    format = what(image_name)
    images_processing.save_image(
        'result_beta={0}_noise_p={1}_iter={2}_neighbors={3}'.format(beta, noise_probability, iterations,
                                                                    neighbors_number), format, sampled_image,
        directory='Results')
    print('success')


if __name__ == '__main__':
    arguments = sys.argv
    image_name, beta, iterations, noise_probability, neighbors_number = arguments[1:]
    denoising_pipeline(image_name, float(beta), int(iterations), float(noise_probability), int(neighbors_number))
