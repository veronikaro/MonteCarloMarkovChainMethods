import csv
from os import path, stat


def is_file_empty(file_path):
    """ Check if file is empty by confirming if its size is 0 bytes"""
    return path.exists(file_path) and stat(file_path).st_size == 0


# TODO: add an ability to save other parameters as well (type of noise, probability of noise, etc)
def write_to_csv_file(filename, metrics_dict, metric_names):
    """
    Write measurements results to the CSV file.

    :param filename: a name of the file to save the metrics to. created if does not exist
    :param metrics_dict: a dictionary of key-value pairs where key is the value of the variable on the X axis, value - the value of the variable on the Y axis
    :param metric_names: a tuple of paramaters names for the header row
    :return: None
    """
    with open(r'{0}'.format(filename), 'a', newline='') as f:
        writer = csv.writer(f)
        if is_file_empty(filename):
            writer.writerow(list(metric_names))
        for key, value in metrics_dict.items():
            writer.writerow([key, value])
