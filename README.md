# Implementation Multilayer Perceptron (MLP)

These are my condensed notes on the basic theory of neural networks. It served for me as a foundation for implementing a neural network properly. Please check the reference section for books and articles I was using as learning material.

## Overview
A MLP is a mathematical function

![ffnn_formula](./readme_files/1.png)

FFNN stands for Feed-forward neural network. `w` are the weights of the connections and `b` are the biases.

![ffnn_drawing](./readme_files/12.png)

The neural net has an input layer representing the values of the feature vector and an output layer for the result value(s) and zero or more hidden layers. All neurons (nodes) are connected.

Weights initialization:

![weights_init](./readme_files/11.png)

`n` is the number of the incoming connections to a neuron.

Output value calculation:

![output_value](./readme_files/2.png)

alpha is the activation function which is often the logistic function or the hyperbolic tangent.

![logistic_func](./readme_files/3.png)

![tanh_func](./readme_files/4.png)

## Training

We feed the neural network with the feature vector of the samples from the training set and compare the result at the output layer with the corresponding test vector. The error is backpropagated in order to update the weights (and biases).

## Gradient descent

To know how to update the `w` and `b` parameters the Gradient descent method is used.

The goal is to minimize the measured error by optimizing the `w` and `b` parameters. To know how to update the parameters the **Gradient descent** method is used.

As the loss function the mean squared error (MSE) function divided by two is selected.

![msq](./readme_files/5.png)

Applied to a neuron `j` it is

![msq_neuron_j](./readme_files/6.png)

The gradient is calculated by determining the partial derivations of `w` and `b` of the error function.

![gradient_descent](./readme_files/7.png)

Not so obvious is what the error of an inner neuron could be because only for the output neurons target values exist. So that's why there are two cases for calculating delta in the above formula.

## Parameter adjustments

Update weight:

![update_w](./readme_files/8.png)

Update bias:

![update_b](./readme_files/10.png)

![learning_rate](./readme_files/9.png)

Note that the logistic activation function is used here as an example. If tanh was selected as the sigmoid function, then the derivative of tanh must be used instead.

## Hyperparameters

So now we know how a multilayer perceptron works. But there are also some parameters we have to define from outside:
- number of layers
- number of nodes of the hidden layers
- learning rate
- activation function
- epoch count

... and more.

There are some methods that may help to select the hyperparameters but it also involves experimenting with a smaller training and validation set.

## Implementation

TODO

## Literature Reference

 * [The Hundred-Page Machine Learning Book](http://themlbook.com/)
 * [Make Your Own Neural Network](https://makeyourownneuralnetwork.blogspot.com/)
 * [Finding the derivative of the error](https://en.wikipedia.org/wiki/Backpropagation#Finding_the_derivative_of_the_error)
