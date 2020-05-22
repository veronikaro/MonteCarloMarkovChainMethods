# Simulation Results

* Analysis of the Metropolis sampler for the noise Ising model

Original picture:

![](https://github.com/veronikaro/MonteCarloMarkovChainMethods/blob/master/MCMC/Binary%20images/bw_grumpy_cat.jpg)

Noised with 10% of flip noise:

![](https://github.com/veronikaro/MonteCarloMarkovChainMethods/blob/master/MCMC/Noisy%20images/noised_10.0%25_grumpy_cat.jpg)

Denoised versions: 

* beta = 0.8, noise_p = 0.1, iterations = 1687500 (overall number of pixels), neighbors = 8 (with diagonals):

![](https://github.com/veronikaro/MonteCarloMarkovChainMethods/blob/master/MCMC/Results/result_beta%3D0.8_noise_p%3D0.1_iter%3D1687500_neighbors%3D8.jpeg)

* beta = 1.6 (other parameters are the same as in the previous case)

![](https://github.com/veronikaro/MonteCarloMarkovChainMethods/blob/master/MCMC/Results/result_beta%3D1.6_noise_p%3D0.1_iter%3D1687500_neighbors%3D8.jpeg)


# Metrics visualization

## Beta values vs the percentage of wrong pixels:

![](https://github.com/veronikaro/MonteCarloMarkovChainMethods/blob/master/Images%20%26%20results/Graphs/beta_vs_wrong_pixels_percentage_noise_p%3D0.1_iter%3D1687500_neighbors%3D8.png)

(probability of noise = **0.1**, iterations = 1687500, neighbors = 8)

![](https://github.com/veronikaro/MonteCarloMarkovChainMethods/blob/master/Images%20%26%20results/Graphs/beta_vs_wrong_pixels_percentage_noise_p%3D0.2_iter%3D1687500_neighbors%3D8.png)

(probability of noise = **0.2**, iterations = 1687500, neighbors = 8)

![](https://github.com/veronikaro/MonteCarloMarkovChainMethods/blob/master/Images%20%26%20results/Graphs/beta_vs_wrong_pixels_percentage_noise_p%3D0.3_iter%3D1687500_neighbors%3D8.png)

(probability of noise = **0.3**, iterations = 1687500, neighbors = 8)

![](https://github.com/veronikaro/MonteCarloMarkovChainMethods/blob/master/Images%20%26%20results/Graphs/beta_vs_wrong_pixels_percentage_noise_p%3D0.4_iter%3D1687500_neighbors%3D8.png)

(probability of noise = **0.4**, iterations = 1687500, neighbors = 8)

These graphs allow us to see how the percentage of wrong pixels is decreasing (though this can't be easily seen on the image)


## Beta values vs PSNR:

![](https://github.com/veronikaro/MonteCarloMarkovChainMethods/blob/master/Images%20%26%20results/Graphs/beta_vs_psnr_noise_p%3D0.1_iter%3D1687500_neighbors%3D8.png)

(probability of noise = **0.1**, iterations = 1687500, neighbors = 8)


![](https://github.com/veronikaro/MonteCarloMarkovChainMethods/blob/master/Images%20%26%20results/Graphs/beta_vs_psnr_noise_p%3D0.2_iter%3D1687500_neighbors%3D8.png)

(probability of noise = **0.2**, iterations = 1687500, neighbors = 8)

![](https://github.com/veronikaro/MonteCarloMarkovChainMethods/blob/master/Images%20%26%20results/Graphs/beta_vs_psnr_noise_p%3D0.3_iter%3D1687500_neighbors%3D8.png)

(probability of noise = **0.3**, iterations = 1687500, neighbors = 8)

![](https://github.com/veronikaro/MonteCarloMarkovChainMethods/blob/master/Images%20%26%20results/Graphs/beta_vs_psnr_noise_p%3D0.4_iter%3D1687500_neighbors%3D8.png)

(probability of noise = **0.4**, iterations = 1687500, neighbors = 8)

These graphs allow us to see how PSNR values are increasing with increasing beta values. As well as evaluate the smoothness of curves for different noise percentages (?)



