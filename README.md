## Image restoration using Markov random field models
This repository holds implementation of Markov chain Monte Carlo methods, namely Gibbs Sampler and Metropolis Sampler. Its purpose is to show how to combine Bayesian approach, Markov random field models, and the Ising Model for restoring monochrome (black & white) images from noise

## Notes on using samplers
1. Convert an image to black & white
2. Add a noise to it
3. Reduce the number of channels from 3 to 1
4. Convert the image to the Ising model representation


## Considered types of noise 
* Flip noise with p parameter
* Additive Gaussian noise with sigma parameter
