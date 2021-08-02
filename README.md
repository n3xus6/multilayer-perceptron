# Implementation Multilayer Perceptron (MLP)

These are my condensed notes on the basic theory of neural networks. It served for me as a foundation for implementing a neural network. Check the reference section for the books and articles I was using as learning material.

## Overview
A MLP is a mathematical function

![ffnn_formula](./readme_files/1.png)

FFNN stands for Feed-forward neural network. `w` are the weights (weight matrices) of the connections and `b` are the biases added to the input of the activation function.

![ffnn_drawing](./readme_files/12.png)

The neural net has an input layer representing the values of the feature vector and an output layer for the result value(s) and zero or more hidden layers. All neurons (nodes) are connected.

Output function for a neuron `j`:

![output_value](./readme_files/2.png)

alpha is the activation function which is often the logistic function

![logistic_func](./readme_files/3.png)

or the hyperbolic tangent

![tanh_func](./readme_files/4.png)

## Training

We feed the neural network with the feature vector of all samples in the training set and compare the result at the output layer with the corresponding test vector for each sample. The error values are backpropagated from one layer to the previous layer and used to update the weights between any two layers of the neural net. When all samples are processed, this step is repeated and repeated again, by randomly picking the samples from the same training set, until some finish criteria is met e.g., the classification accuracy rating was at it's peak. Data preprocessing and adding artificial training data could also improve the model.

Weight initialization:

![weights_init](./readme_files/11.png)

`n` is the number of the incoming connections (signals) to a neuron.

This is just one example. Other methods exist.

## Gradient descent

The goal is to minimize the error of the neural net by optimizing the `w` and `b` parameters in the error function. To know how to update the parameters, the **Gradient descent** method is applied.

As the loss function the mean squared error (MSE) divided by two is used here.

![msq](./readme_files/5.png)

For a neuron `j` it is

![msq_neuron_j](./readme_files/6.png)

The gradient is calculated by determining the partial derivations for `w` and `b` of the selected error function.

![gradient_descent](./readme_files/7.png)

Not so obvious is what the error of an inner neuron could be, because only for the output neurons target values exist. So that's why there are two cases for calculating delta in the above formula.

![gradient_descent](./readme_files/13.png)

## Parameter adjustments

Assuming the logistic function was selected as the sigmoid function:

Update weight:

![update_w](./readme_files/8.png)

Update bias:

![update_b](./readme_files/10.png)

![learning_rate](./readme_files/9.png)

## Hyperparameters

So we know how a multilayer perceptron works, now. But there are also some parameters that we have to specify from outside, that are not part of the learning process:
- number of layers
- number of nodes on the hidden layers
- learning rate
- activation function
- weight (bias) initialization
- batch size
- epoch count

... and more.

There are some methods that help to select good hyperparameters but it also involves experimenting with a smaller training and validation set.

## Implementation

TODO

## Literature Reference

 * [The Hundred-Page Machine Learning Book](http://themlbook.com/)
 * [Make Your Own Neural Network](https://makeyourownneuralnetwork.blogspot.com/)
 * [Finding the derivative of the error](https://en.wikipedia.org/wiki/Backpropagation#Finding_the_derivative_of_the_error)
