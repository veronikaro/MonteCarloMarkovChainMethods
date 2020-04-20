import numpy as np
import timeit
import random
import cython
from math import exp, log
from MCMC.metropolis_sampler import clique_energy, restore_channels, reduce_channels_for_sampler, \
    convert_image_to_ising_model, convert_from_ising_to_image
import cv2

if __name__ == '__main__':

    print("hello, I'm main")
