/* Copyright (c) 2021 A. Heinemann. All rights reserved.
 * For Personal Use only. No guarantees. Use at your own risk.
 *
 * Multilayer perceptron (MLP) implementation.
 *
 * Homepage: https://github.com/n3xus6/multilayer-perceptron
 */
#pragma once

 /* Min-max normalization. */
#define RESCALE(X, MIN_X, MAX_X, A, B) \
        ((A) + (((X) - (MIN_X)) * ((B) - (A))) / ((MAX_X) - (MIN_X)))

enum activation_func_id {
    SIGMOID_LOGISTIC = 0, /* Logistic function.  */
    SIGMOID_TANH,         /* Hyperbolic tangent. */
};

struct hyperparams {
    const char *layout;              /* Neural network layout "N1;N2;N3;...", N: nbr. of nodes per layer. */
    double learn;                    /* Learning rate. */
    enum activation_func_id func_id; /* Activation function id. */
};

struct neuralnet;

struct vecd {
    int len;
    double attrs[];
};

/* Creates a neural network.
 */
struct neuralnet *neuralnet_create(struct hyperparams *params);

/* Trains the neural network with the feature vector and the label of one sample at a time.
 *  feature - Feature vector with normalized attributes.
 *  label   - Label (target) vector with normalized values.
 *  Returns zero on success.
 */
int neuralnet_train(struct neuralnet *nn, struct vecd *feature, const struct vecd *label);

/* Queries the neural network for the given feature vector.
 *  feature - Feature vector with normalized attributes.
 *  label   - Label output vector from the neuralnet.
 *  Returns zero on success.
 */
int neuralnet_query(struct neuralnet *nn, struct vecd *feature, struct vecd *label);

/* Withdraws the neural network.
 */
void neuralnet_free(struct neuralnet *nn);
