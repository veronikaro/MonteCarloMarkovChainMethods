from PIL import Image
from matplotlib import image
from matplotlib import pyplot
import cv2
import numpy as np
import random


def sp_noise(image, prob):
    """
        Add a salt and pepper noise to the given image

        :param image: image to add the noise to
        :param prob: probability of the noise
        :return: image degraded by noise
        """
    output = np.zeros(image.shape, np.uint8)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output


def test_sp_noise(image_path):
    image = cv2.imread(image_path, 0)
    noise_img = sp_noise(image, 0.2)  # only 5% of noise is added. You can vary the percentage of added noise
    filename = image_path[7:-4] + '_noisy'
    save_image(filename, noise_img)
    cv2.imshow('Noised image', noise_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Save image using OpenCV library
def save_image(filename, format, image):
    cv2.imwrite(filename + '.' + format, image)


if __name__ == '__main__':
    test_sp_noise('Images/cat_bw.png')


def add_gaussian_noise(image_in, noise_sigma):
    temp_image = np.float64(np.copy(image_in))

    h = temp_image.shape[0]
    w = temp_image.shape[1]
    noise = np.random.randn(h, w) * noise_sigma

    noisy_image = np.zeros(temp_image.shape, np.float64)
    if len(temp_image.shape) == 2:
        noisy_image = temp_image + noise
    else:
        noisy_image[:, :, 0] = temp_image[:, :, 0] + noise
        noisy_image[:, :, 1] = temp_image[:, :, 1] + noise
        noisy_image[:, :, 2] = temp_image[:, :, 2] + noise

    """
    print('min,max = ', np.min(noisy_image), np.max(noisy_image))
    print('type = ', type(noisy_image[0][0][0]))
    """

    return noisy_image