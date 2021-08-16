/* Copyright (c) 2021 A. Heinemann. All rights reserved.
 * For Personal Use only. No guarantees. Use at your own risk.
 *
 * Multilayer perceptron (MLP) implementation.
 *
 * Homepage: https://github.com/n3xus6/multilayer-perceptron
 *
 * Books and articles:
 * "The Hundred-Page Machine Learning Book", http://themlbook.com/.
 * "Make Your Own Neural Network", https://makeyourownneuralnetwork.blogspot.com/.
 * "Finding the derivative of the error", https://en.wikipedia.org/wiki/Backpropagation#Finding_the_derivative_of_the_error.
 */
#include "neuralnet.h"

#include <math.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct node {
	double out;
	double delta;
	double bias;
};

struct vector {
	int len;
	union {
		struct node *nodes;
		double *values;
	};
};

static double f_logistic(double x) { return 1 / (1 + exp(-x)); }
static double f_tanh(double x)     { return tanh(x); }
static double d_logistic(double y) { return y * (1 - y); }
static double d_tanh(double y)     { return y * (1 / y - y); }

/* Maps a random value [0, RAND_MAX] to [-1/Sqrt(N), 1/Sqrt(N)]. */
#define INIT_VAL(N) RESCALE((double)rand(), 0, RAND_MAX, -1 / sqrt((N)), 1 / sqrt((N)))

typedef double (*activation_fp)(double);

static const struct {
	activation_fp function;
	activation_fp derivative;
} activation[] = {
	{ f_logistic, d_logistic },
	{ f_tanh, d_tanh},
};

struct neuralnet
{
	struct vector *layers;
	int nlayers;
	struct vector *input;
	struct vector *output;
	struct vector *weights;
	int nweights;
	enum activation_func_id func_id;
	double learn;
};

static int create_layers(struct neuralnet *, const char *);
static int create_weights(struct neuralnet *);
static void free_vectors(struct vector *, int);
static void feedforward(struct neuralnet *, const struct vecd *);
static void backpropagate(struct neuralnet *, const struct vecd *);

struct neuralnet *neuralnet_create(struct hyperparams *params)
{
	if (!params || !params->layout)
		return NULL;

	struct neuralnet *nn = calloc(1, sizeof(struct neuralnet));
	if (!nn)
		return NULL;

	if (create_layers(nn, params->layout) || create_weights(nn)) {
		neuralnet_free(nn);
		return NULL;
	}

	nn->learn = params->learn;
	nn->func_id = params->func_id;

	srand((unsigned int)time(0));

	return nn;
}

void neuralnet_free(struct neuralnet *nn)
{
	if (nn) {
		free_vectors(nn->layers, nn->nlayers);
		free_vectors(nn->weights, nn->nweights);
		free(nn);
	}
}

int neuralnet_train(struct neuralnet *nn, struct vecd *feature, const struct vecd *label) {
	if (!nn || !feature || feature->len != nn->input->len || !label || label->len != nn->output->len)
		return -1;

	feedforward(nn, feature);
	backpropagate(nn, label);

	return 0;
}

int neuralnet_query(struct neuralnet *nn, struct vecd *feature, struct vecd *label) {
	if (!nn || !feature || feature->len != nn->input->len || !label || label->len != nn->output->len)
		return -1;

	feedforward(nn, feature);

	for (int i = 0; i < label->len; i++)
		label->attrs[i] = nn->output->nodes[i].out;

	return 0;
}

static void free_vectors(struct vector *vectors, int n)
{
	if (vectors && n > 0) {
		while (n--)
			if (vectors[n].nodes)
				free(vectors[n].nodes);
		free(vectors);
	}
}

static int create_layers(struct neuralnet *nn, const char *layout) {
	int layer_count = 1;
	for (int i = 0; layout[i]; i++)
		if (layout[i] == ';')
			layer_count++;

	if (layer_count < 2)
		return -1;

	if (!(nn->layers = calloc((size_t)layer_count, sizeof(struct vector))))
		return -1;

	nn->nlayers = layer_count;

	char *buf = calloc(strlen(layout) + 1, sizeof(char));
	if (!buf)
		return -1;

	strcpy(buf, layout);

	char *p = strtok(buf, ";");
	for (int i = 0; p; p = strtok(NULL, ";"), i++) {
		int n = 0;

		if (sscanf(p, "%d", &n) != 1 ||	n < 1 || i == layer_count ||
			!(nn->layers[i].nodes = calloc((size_t) n, sizeof(struct node)))) {
			free(buf);
			return -1;
		}

		nn->layers[i].len = n;

		if (i > 0) {
			for (int j = 0; j < nn->layers[i].len; j++)
				nn->layers[i].nodes[j].bias = INIT_VAL(nn->layers[i - 1].len);
		}
	}

	nn->input = &nn->layers[0];
	nn->output = &nn->layers[nn->nlayers - 1];

	free(buf);

	return 0;
}

static int create_weights(struct neuralnet *nn) {
	if (!(nn->weights = calloc((size_t)nn->nlayers - 1, sizeof(struct vector))))
		return -1;

	nn->nweights = nn->nlayers - 1;

	for (int i = 0; i < nn->nweights; i++)	{
		nn->weights[i].len = nn->layers[i].len * nn->layers[i + 1].len;
		if (!(nn->weights[i].values = calloc((size_t)nn->weights[i].len, sizeof(double))))
			return -1;
		for (int j = 0; j < nn->weights[i].len; j++)
			nn->weights[i].values[j] = INIT_VAL(nn->layers[i].len);
	}

	return 0;
}

#define WEIGHT_AT(L, J, K) nn->weights[(L)].values[nn->layers[(L)].len * (K) + (J)]

/* Feed-forward operation.
 */
static void feedforward(struct neuralnet *nn, const struct vecd *feature)
{
	/* The input layer just holds the feature attributes. */
	for (int i = 0; i < feature->len; i++)
		nn->input->nodes[i].out = feature->attrs[i];

	/* Output of neuron j: 'o_j = sigmoid(sum_i(w_ij * o_i) + b_j)' */
	for (int p = 0, q = 1; q < nn->nlayers; p++, q++) {
		for (int j = 0; j < nn->layers[q].len; j++) {
			nn->layers[q].nodes[j].out = 0;
			for (int i = 0; i < nn->layers[p].len; i++)
				nn->layers[q].nodes[j].out += nn->layers[p].nodes[i].out * WEIGHT_AT(p, i, j);
			nn->layers[q].nodes[j].out = activation[nn->func_id].function(nn->layers[q].nodes[j].out + nn->layers[q].nodes[j].bias);
		}
	}
}

/* Backpropagation.
 *
 * The loss function 'L = 1/2n*sum_x(t_x - y_x)^2' shall be optimized.
 * For a neuron j it is 'E=1/2*(t_j - o_j)^2' with 'o_j = sigmoid(sum_i(o_i * w_ij) + b_j)'.
 * Computing the derivative dE/dw_ij, which is '-(delta_j) * dsigmoid/d_wij * o_i', with 'delta_j = t_j - o_j'.
 * For inner neuron i, t_i is unknown. In this case 'delta_i = sum_j(w_ij * delta_j)'.
 * Finally, the weight update is 'w_ij_new = w_ij_old - a * dE/dw_ij', with 'a' the learning rate.
 * The biases are updated similarly by the expression dE/db_j.
 */
static void backpropagate(struct neuralnet *nn, const struct vecd *test) {

	/* Output layer: just have to set the error values. */
	for (int j = 0; j < nn->output->len; j++)
		nn->output->nodes[j].delta = nn->output->nodes[j].out - test->attrs[j];

	/* Hidden layers: have to compute the error of the output values and update the weight matrix. */
	for (int p = nn->nlayers - 2, q = p + 1; p > 0; p--, q--) {
		for (int i = 0; i < nn->layers[p].len; i++) {
			double delta = 0;
			for (int j = 0; j < nn->layers[q].len; j++) {
				delta += WEIGHT_AT(p, i, j) * nn->layers[q].nodes[j].delta;
				WEIGHT_AT(p, i, j) -= nn->learn * nn->layers[q].nodes[j].delta * activation[nn->func_id].derivative(nn->layers[q].nodes[j].out) * nn->layers[p].nodes[i].out;
			}
			nn->layers[p].nodes[i].delta = delta;
		}
	}

	/* Input layer: just have to update the weight matrix. */
	for (int i = 0; i < nn->input->len; i++) {
		for (int j = 0; j < nn->layers[1].len; j++)
			WEIGHT_AT(0, i, j) -= nn->learn * nn->layers[1].nodes[j].delta * activation[nn->func_id].derivative(nn->layers[1].nodes[j].out) * nn->input->nodes[i].out;
	}

	/* For all but the input layer the bias values are updated. */
	for (int l = 1; l < nn->nlayers; l++) {
		for (int j = 0; j < nn->layers[l].len; j++) {
			nn->layers[l].nodes[j].bias -= nn->learn * nn->layers[l].nodes[j].delta * activation[nn->func_id].derivative(nn->layers[l].nodes[j].out);
		}
	}
}
