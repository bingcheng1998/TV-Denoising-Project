- modify your previous code with stochastic strategy for large-scale problems

---

Please check `Optimization_for_Total_Variation_Image_Denoising.pdf` for full report with pictures.

---

## Implement the Stochastic Method to Accelerate Gradient Descent

Because there are many pixels in one single image, it would be difficult to calculate the best descent direction, and for one direction, the step size would not be the best length for every pixel.

To make the selection of step size more efficient, it could be better to take apart the single image to multiple small images, and then do gradient descent for every small image separately.

A simple solution is to split an image to multiple n by n rectangles, and according to figure \ref{img5}, there should be a margin as a helper to calculate the differentiation of the center split images. 

To make the process simpler and easy to understand, we define one iteration as one full polling of all split blocks. And there will always process all blocks one by one with no multiprocess, which means in this case, the time would be the real time that needed to finish all the calculations without any trick.

First, let me to introduce you the $f(x^{k})$ vs. time and  $f(x^{k})$ vs. iteration k are presented to show the speed of convergence, in figure \ref{img16} and figure \ref{img17}, compared with the general gradient discent and the gradient discent accelarated by Nesterov method.

[img17]
Comparision of $f(x^{k})$ vs. time for gradient descent, gradient descent accelerated by the Nesterov method, and gradient descent accelerated by the Stochastic method.

Comparision of $f(x^{k})$ vs. iteration k for gradient descent, gradient descent accelerated by the Nesterov method, and gradient descent accelerated by the Stochastic method.

As you can see in figure \ref{img16}, the overall time of 100 iterations for Gradient descent with the stochastic method is longer than the previous two methods, this is because there need to calculate the best step size with exact line search for $(k/n)^2$ times, where $k$ is the length of the image, and $n$ is the size of the block you choose. Though the calculation of the best step size is much more than the previous two solutions, it works better when time is long enough. This is because when some block with a low score has already reached its best state, there can be many other blocks that can be improved largely without being affected by those already converged blocks.

Considering figure \ref{img16}, we can get the same conclusion, but due to the inconsistency of the time and iteration of the stochastic method, it’s better to consider only the  $f(x^{k})$ vs. time figure.

Second, How will the block size $n$ affect the speed of convergence? Here the size $n$ means the block is $n \times n$ block. The block size 2, 4, 8, 16 are considered in the experiemnt. let me to introduce you the $f(x^{k})$ vs. time and  $f(x^{k})$ vs. iteration k are presented to show the speed of convergence, in figure \ref{img18} and figure \ref{img19},

[figures]

Comparision of $f(x^{k})$ vs. time for gradient descent accelerated by the Stochastic method with block size 2, 4, 8, 16.

Comparision of $f(x^{k})$ vs. iteration k for gradient descent accelerated by the Stochastic method with block size 2, 4, 8, 16.

Considering the  $f(x^{k})$ vs. iteration k figure, it can be found that the speed of convergence increased as the size of block decreased. This is because the smaller the block size, the easier the best step size can reach its minimal value, and then the faster the image to fit its denoised state. However, looking at figure  \ref{img18}, it will be found that the overall time for 100 iterations is very different for those block sizes. 

According to figure \ref{img18}, the speed of convergence is fastest for block size 2, but the overall time for block sez 2 is the longest, which is almost the double of block size 4, or three times longer than the block size 8. The block size 4 will cause the shortest overall time, and enlarge or reduce the size of the block will both cause the overall time to be longer.

For all the previous experiments, the number of $\lambda$ is 0.9. This is because for gradient descent and gradient descent accelerated by the Nesterove method are both slow, which means for small $\lambda$ like 0.1, 0.2, 0.5, the noised image cannot converge as fast as the user wants. But for the stochastic method, the speed of convergence is fast, which causes the noised image being over-denoised, as shown in figure \ref{img20}.

[img]
Figure 20. The original image, noised image, and three images are the denoised image with iteration 1, 10, and 100 with gradient descent accelerated by the Stochastic method with block size 4, 8, and 16.

As you can see, as the decrease of block size, the noised image being smother faster, but for block size 8 and 4, they are over-denoised. In this case, what we need to do is to decrease $\lambda$, and by experiment, $\lambda=0.1$ works well for block size 8 and 4.