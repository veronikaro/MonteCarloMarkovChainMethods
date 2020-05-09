import numpy as np
import matplotlib.pyplot as plt


# Visualization module


# Remember what beta is. beta = 1 / T
# Instead of beta, we can start with T

# TODO
def beta_vs_wrong_pixels(beta_list, errors_list, noise_type, noise_level, sampler_type=''):
    """
    Build a plot with beta values vs wrong pixels percentage and save it to the file. Specify the level of initial noise and a type of sampler used.

    :param beta_list: a list of beta values which were used in different algorithm runs
    :param errors_list: a list of wrong pixels percentages corresponding to the beta values
    :param noise_type: a type of noise by which the image has been degraded
    :param noise_level: a noise level (expressed as probability if noise_type='flip' or as a sigma value is noise_type='gaussian')
    :return: a pyplot object
    """
    pass


def save_graph_to_file(plot, directory):
    """
    Save the plot to the given directory.

    :param plot: a graph
    :param directory: a path to the file
    :return: None
    """
    pass
