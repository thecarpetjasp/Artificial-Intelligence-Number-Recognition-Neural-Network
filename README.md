# Artificial Intelligence for Number Recognition
## About

This is a neural network created from scratch by Jack Carter.

This model can correctly predict hand drawn numbers from images. It was trained on a sample of 1,300+ images with a 100% accuracy and a loss value of 0.01.

# Samples
Samples used for training were hand drawn pictures of numbers ranging from 0-9 on a 28x28 pixel resolution.

Here is an example of these samples:

![alt text](https://upload.wikimedia.org/wikipedia/commons/f/f7/MnistExamplesModified.png)

# Reliability
The model was trained in approximately 60 seconds achieving an accuracy of 100% and a loss value of 0.01. With more tweaks and changes to the learning rate, decay and momentum of the Optimizer SGD, you could easily lower the loss further.

# Specifications
The model consists of a vanilla neural network using the Feed-Forward method along with the Rectified Linear Unit activation function and the Softmax activation function. The model uses backpropagation with gradient descent to determine the derivates of the weights
and biases in relation to the loss value. An optimizer using Stochastic Gradient Descent is then used to change these weights and biases accordingly.
* Feed-Forward network
* Activation Functions:
   - Rectified Linear Unit
   - Softmax
* Optimizer:
  - Stochastic Gradient Descent

# Notes
This model could be trained on a lot more samples. I avoided using any existing datasets such as the [MNIST database](https://en.wikipedia.org/wiki/MNIST_database) as I wanted to build everything from the ground up including training data.

However, even with the 1,300+ samples I created myself, the model showed very consistent results, and was able to successfully predict 100% of all custom validation datasets provided to it.

This model is not anything groundbreaking in the world of Artificial Intelligence. But, I hope this could serve someone well, who might be looking for some insights on how neural networks can be built from scratch without using frameworks such as PyTorch or Tensorflow.


# Extra
I've also included a neural network trained on the [Kannada MNIST Dataset](https://github.com/vinayprabhu/Kannada_MNIST).

Instead of using English numerals, the **Kannada MNIST Dataset** uses Kannada digits. Kannada is a language spoken predominantly by people of Karnataka in southwestern India. The language has roughly 45 million native speakers and is written using the Kannada script. [Wikipedia](https://en.wikipedia.org/wiki/Kannada)

![alt text](https://storage.googleapis.com/kaggle-media/competitions/Kannada-MNIST/kannada.png)
