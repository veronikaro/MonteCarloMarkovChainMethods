## Image restoration using Markov random field models
This repository holds the implementation of Markov chain Monte Carlo methods, namely Gibbs Sampler and Metropolis Sampler. Its purpose is to show how to combine the Bayesian approach, Markov Random Field models, and the Ising Model for restoring monochrome (black & white) images from noise.  

#### Thoughts: 
Denoising techniques can be especially beneficial for real-time computer vision applications & improving performance of neural networks that deal with classification & detection

## Notes on using the samplers
1. Convert an image to its black & white version
2. Add noise to it using noiser.py
3. Reduce the number of channels from 3 to 1
4. Convert the image to the Ising model representation


## Considered types of noise 
* Flip noise with the parameter **p** - a probability of the pixel being written wrong
* Additive Gaussian noise with the **sigma** parameter

## Scripts that can be run separately from the command line:

* images_processing.py (for converting the image from a current directory to its b&w version and adding noise to it)
* metropolis_ising_with_noise.py (run algorithm & save results for a given image)
* metropolis_ising_no_noise.py

Note that Python 3.x must be installed on your machine.

## How to run MCMC/images_processing.py:
Run this script from the command line to get two versions of an image in separate folders (`/Binary images` and `/Noisy images`. The folders are not created if already exist):
1. Black & white
2. Noisy

*Arguments*:

* original_image_name - the absolute or relative path to the original image (can be colored or greyscale)
* noise_probability - the probability of flip noise to add to the image

*Example of running the script*:

`> python images_processing.py grumpy_cat.jpg 0.5`


## How to run MCMC/denoiser.py:

Saves the results of denoising to the folder 'Results' (it's created if doesn't exist)

*Arguments*:

* image_name - the absolute or relative path to the noised black & white image
* beta - an essential parameter of the sampler, an inverse to the temperature (1/T) (any float number)
* iterations - the number of iterations
* noise_probability - the assumed probability of the flip noise in the image
* neighbors_number: the size of the neighborhood structure. it's used when calculating the energy of the neighborhood. the default is 8. can be switched to 4

*Example of running the script*:

`> python denoiser.py brain4_noise5%.jpg 0.8 100000 0.05 8`


