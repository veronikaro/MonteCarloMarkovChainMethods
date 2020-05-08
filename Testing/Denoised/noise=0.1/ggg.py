import numpy as np
import cv2

from MCMC import images_comparison

if __name__ == '__main__':
    im1 = cv2.imread('noised_chest10%.jpeg')
    im2 = cv2.imread('metropolis_noise_ising_chest_b=1.3_noise=0.1.jpeg')
    rows = range(im1.shape[0])
    columns = range(im1.shape[1])
    print(im1.shape[0]*im1.shape[1])
    wrong_pixels = 0

    for i in rows:
        for j in columns:
            if im1[i, j].any() != im2[i, j].any():
                wrong_pixels += 1
    print(wrong_pixels)
    #print(images_comparison.mse(im1, im2))
    #print(np.sum(np.absolute(im1 - im2)) / (im1.shape[0]*im2.shape[1]) / 255)