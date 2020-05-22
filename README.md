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
