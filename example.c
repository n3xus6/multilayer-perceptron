/* Example code showing the use of "neuralnet", a multilayer perceptron implementation.
 * Homepage: https://github.com/n3xus6/multilayer-perceptron
 */
#include "neuralnet.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>

struct idx {
	unsigned char *hdr;
	size_t hdr_len;
	unsigned char *data;
	size_t data_len;
};

static struct idx *load_idx(const char *path);
static void unload_idx(struct idx *idx);

#define GET_DIMENSION(IDX, D)           ( \
	(IDX)->hdr[((D) + 1) * 4]     << 24 | \
	(IDX)->hdr[((D) + 1) * 4 + 1] << 16 | \
	(IDX)->hdr[((D) + 1) * 4 + 2] << 8  | \
	(IDX)->hdr[((D) + 1) * 4 + 3]       )

struct dataset {
	struct idx *labels;
	struct idx *samples;
};

/* The error handling is not exhaustive in this example. */
int main(void)
{
	/* Fashion-MNIST dataset: https://github.com/zalandoresearch/fashion-mnist */
	const struct dataset training = {
		.labels = load_idx("train-labels-idx1-ubyte"),
		.samples = load_idx("train-images-idx3-ubyte"),
	};

	const struct dataset testing = {
		.labels = load_idx("t10k-labels-idx1-ubyte"),
		.samples = load_idx("t10k-images-idx3-ubyte"),
	};

	if (!training.labels || !training.samples || !testing.labels || !testing.samples)
		return -1;

	const int attributes_count = GET_DIMENSION(training.samples, 1) * GET_DIMENSION(training.samples, 2);
	const int train_samples_count = GET_DIMENSION(training.samples, 0);
	const int test_samples_count = GET_DIMENSION(testing.samples, 0);
	const int classes_count = 10;

	/* Specify the hyperparameters.
	 * Finding the right parameters and network layout can be a difficult task.
	 * A small change to one of these values can have a large impact. */
#if 1
	const enum activation_func_id func_id = SIGMOID_LOGISTIC;
	const double learning = 0.10;
#else
	const enum activation_func_id func_id = SIGMOID_TANH;
	const double learning = 0.001;
#endif

	/* Create artificial neural network (ANN) with one hidden layer. */
	char layout[64];
	sprintf(layout, "%i;%i;%i", attributes_count, attributes_count / 2, classes_count);
	struct neuralnet *nn = neuralnet_create(&(struct hyperparams) {
		.layout = layout,
		.func_id = func_id,
		.learn = learning
	});

	struct vecd *feature_vec = malloc(sizeof(struct vecd) + attributes_count * sizeof(double));
	if (!feature_vec) return -1;
	feature_vec->len = attributes_count;

	struct vecd *label_vec = malloc(sizeof(struct vecd) + classes_count * sizeof(double));
	if (!label_vec) return -1;
	label_vec->len = classes_count;
	
	srand((unsigned int)time(NULL));

	/* The ANN is fed with the training data repeadingly, where the order in which the
	   samples are selected is randomly chosen. The classification accuracy rate,
	   after increasing, can drop and increasing again to an higher value than before. */
	const int epoch = 32;
	for (int e = 0; e < epoch; e++) {

		/* Model training. */
		int rand_start = rand() % (train_samples_count - 1);
		for (int i = rand_start + 1; i != rand_start; i = (i + 1) % train_samples_count) {

			for (int j = 0; j < attributes_count; j++)
				feature_vec->attrs[j] = RESCALE((double) training.samples->data[i * attributes_count + j], 0, 255, 0, 1);

			for (int j = 0; j < classes_count; j++)
				label_vec->attrs[j] = 0.01;

			label_vec->attrs[training.labels->data[i]] = 0.99;

			neuralnet_train(nn, feature_vec, label_vec); /* will take some time */
		}

		/* Model testing. */
		int hits = 0;
		for (int i = 0; i < test_samples_count; i++) {
			const int expected = testing.labels->data[i];

			for (int j = 0; j < attributes_count; j++)
				feature_vec->attrs[j] = RESCALE((double) testing.samples->data[i * attributes_count + j], 0, 255, 0, 1);

			neuralnet_query(nn, feature_vec, label_vec);

			int predicted = 0;
			for (int p = 0; p < classes_count; p++)
				if (label_vec->attrs[p] > label_vec->attrs[predicted])
					predicted = p;

			if (predicted == expected)
				hits++;
		}

		double accuracy = (double) hits / test_samples_count;
		printf("[Epoch %2i] Classification accuracy rate: %f\n", e, accuracy);
	}

	neuralnet_free(nn);

	unload_idx(training.labels);
	unload_idx(training.samples);
	unload_idx(testing.labels);
	unload_idx(testing.samples);
		
	return 0;
}

static struct idx *load_idx(const char *path) {
	FILE *fp;
	struct idx *idx;

	if (!(idx = calloc(1, sizeof(struct idx))))
		return NULL;

	if (!(fp = fopen(path, "rb")))
		return NULL;

	fseek(fp, 3, SEEK_SET);
	int num_dimensions = fgetc(fp);
	if (num_dimensions < 0)
		return NULL;

	idx->hdr_len = (size_t) (num_dimensions + 1) * 4;

	if (!(idx->hdr = malloc(idx->hdr_len)))
		return NULL;

	rewind(fp);
	if (fread(idx->hdr, idx->hdr_len, 1, fp) != 1)
		return NULL;

	idx->data_len = 1;
	for (unsigned char *p = &idx->hdr[4]; p != &idx->hdr[idx->hdr_len]; p += 4)
		idx->data_len *= (p[0] << 24 | p[1] << 16 | p[2] << 8 | p[3]);

	if (!(idx->data = malloc(idx->data_len)))
		return NULL;

	if (fread(idx->data, idx->data_len, 1, fp) != 1)
		return NULL;

	fclose(fp);

	return idx;
}

static void unload_idx(struct idx *idx) {
	if (idx) {
		if (idx->hdr)
			free(idx->hdr);
		if (idx->data)
			free(idx->data);
		free(idx);
	}
}
