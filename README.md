# Visual-Object-Tracking-using-Color-Representation

## Object Tracking Algorithm

### Overview
A Tracking algorithm based on Bayes classifier is proposed in the paper which uses Color representation as feature descriptor to predict target position with 5 degree of freedom

The covariance matrix helps tracker in adjusting with changing shape and size of target object. The algorithm can adjust its candidates to move towards target region. Moreover Filtering algorithm proposed in the report can facilitates the candidates to converge at the target region by accounting the local best candidate and global best candidate in the density function.

### Algorithm Details
Color Histograms: The algorithm uses color histograms to capture the color distribution of foreground and background regions.

Bayesian Classification: A Naive Bayesian classifier calculates the probability of each pixel being part of the foreground.

Particle Filter: The object state is represented as a set of particles, and a Bayesian filter is used to resample and update these particles based on likelihood.

Particle Swarm Optimization: PSO is employed to refine object positions, improving tracking accuracy.

Correction Mechanisms: Techniques, such as covariance-based correction, are implemented to ensure accurate object state estimation.

Drifting Correction: The algorithm corrects for drifting by adjusting particle positions based on their probabilities.

Loss Function: A loss function, based on the area of the background in the image, is used to assess tracking performance.





Below, I have demonstrated the functioning of the object detection model developed using the VOT dataset.


![](https://github.com/soeb-hussain/Visual-Object-Tracking-using-Color-Representation/blob/main/git_utility/jogging.gif)
<!-- ![](https://github.com/soeb-hussain/Visual-Object-Tracking-using-Color-Representation/blob/main/git_utility/tiger.gif) -->
<!-- ![](https://github.com/soeb-hussain/Visual-Object-Tracking-using-Color-Representation/blob/main/git_utility/basketball.gif) -->
<!-- ![](https://github.com/soeb-hussain/Visual-Object-Tracking-using-Color-Representation/blob/main/git_utility/sunshade.gif) -->
