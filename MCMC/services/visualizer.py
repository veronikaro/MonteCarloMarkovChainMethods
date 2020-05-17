import numpy as np
import matplotlib.pyplot as plt
from MCMC import images_comparison
from MCMC.services import auxiliary_methods
import cv2


# import plotly.express as px
# import pandas
# Visualization module


# Remember what beta is. beta = 1 / T
# Instead of beta, we can start with T

# TODO: change the simplified version to a more detailed one
# TODO: make a decorator to be able to pass the metric function as an argument
def beta_vs_metric_value(errors_dict):  # , noise_type, noise_level, sampler_type=''):
    """
    Build a plot with beta values vs wrong pixels percentage and save it to the file. Specify the level of initial noise and a type of sampler used.
    :param errors_dict: a dictionary of beta-error pairs
    :param noise_type: a type of noise by which the image has been degraded
    :param noise_level: a noise level (expressed as probability if noise_type='flip' or as a sigma value is noise_type='gaussian')
    :return: a pyplot object
    """

    lists = sorted(errors_dict.items())  # sort by key, return a list of tuples

    x, y = zip(*lists)  # unpack a list of pairs into two tuples
    plt.xticks(x)
    plt.xlabel('beta values')
    plt.ylabel('PSNR')
    plt.plot(x, y)
    plt.show()


def save_graph_to_file(plot, directory):
    """
    Save the plot to the given directory.

    :param plot: a graph
    :param directory: a path to the file
    :return: None
    """
    pass


if __name__ == '__main__':
    beta_list = auxiliary_methods.arithmetic_progression_series(0.8, 0.1, 10)
    errors_map = dict()
    images_names = ['result_beta=0.8_noise_p=0.2_iter=1687500_neighbors=8.jpeg',
                    'result_beta=0.9_noise_p=0.2_iter=1687500_neighbors=8.jpeg',
                    'result_beta=1.0_noise_p=0.2_iter=1687500_neighbors=8.jpeg',
                    'result_beta=1.1_noise_p=0.2_iter=1687500_neighbors=8.jpeg',
                    'result_beta=1.2_noise_p=0.2_iter=1687500_neighbors=8.jpeg',
                    'result_beta=1.3_noise_p=0.2_iter=1687500_neighbors=8.jpeg',
                    'result_beta=1.4_noise_p=0.2_iter=1687500_neighbors=8.jpeg',
                    'result_beta=1.5_noise_p=0.2_iter=1687500_neighbors=8.jpeg',
                    'result_beta=1.6_noise_p=0.2_iter=1687500_neighbors=8.jpeg',
                    'result_beta=1.7_noise_p=0.2_iter=1687500_neighbors=8.jpeg',]

    original = cv2.imread('../Binary images/bw_grumpy_cat.jpg')
    for beta, im_name in zip(beta_list, images_names):
        denoised = cv2.imread('../Results/{0}'.format(im_name))
        err = images_comparison.percentage_of_wrong_pixels(original, denoised)
        #err = images_comparison.PSNR(original, denoised)
        errors_map[beta] = err
    beta_vs_metric_value(errors_map)
