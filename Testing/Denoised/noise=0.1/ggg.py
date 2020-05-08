import numpy as np
import cv2

from MCMC import images_comparison

if __name__ == '__main__':
    im1 = cv2.imread('noised_chest10%.jpeg')
    im2 = cv2.imread('metropolis_noise_ising_chest_b=1.2_noise=0.1.jpeg')
    error = images_comparison.percentage_of_wrong_pixels(im1, im2)
    print(error)
    # TODO: graph: beta vs percentage of wrong pixels
