import numpy as np
import cv2
from MCMC.auxiliary_methods import arithmetic_progression_series
from MCMC import images_comparison
import matplotlib.pyplot as plt

if __name__ == '__main__':
    original_im = cv2.imread('chest_bw.jpeg')
    sampled_images = [
        'metropolis_noise_ising_chest_b=0.3_noise=0.1.jpeg',
        'metropolis_noise_ising_chest_b=0.4_noise=0.1.jpeg',
        'metropolis_noise_ising_chest_b=0.5_noise=0.1.jpeg',
        'metropolis_noise_ising_chest_b=0.6_noise=0.1.jpeg',
        'metropolis_noise_ising_chest_b=0.7_noise=0.1.jpeg',
        'metropolis_noise_ising_chest_b=0.8_noise=0.1.jpeg',
        'metropolis_noise_ising_chest_b=0.9_noise=0.1.jpeg',
        'metropolis_noise_ising_chest_b=1.0_noise=0.1.jpeg',
        'metropolis_noise_ising_chest_b=1.1_noise=0.1.jpeg',
        'metropolis_noise_ising_chest_b=1.2_noise=0.1.jpeg',
        'metropolis_noise_ising_chest_b=1.3_noise=0.1.jpeg']
    errors = []
    for im in sampled_images:
        im = cv2.imread(im)
        error = images_comparison.percentage_of_wrong_pixels(im, original_im)
        errors.append(error)
    betas = arithmetic_progression_series(0.3, 0.1, 11)
    plt.plot(betas, errors)
    plt.show()
    # TODO: graph: beta vs percentage of wrong pixels
