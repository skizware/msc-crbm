import rbm_trainer
import numpy as np
import skimage.exposure as exposure
import os


class DbnTrainer(object):
    def __init__(self, dbn, train_set, valid_set, stats_collection_period=500, num_training_epochs=1,
                 output_directory='', image_loader=None, learning_rate_eval_period=25, lr_error_multiplier=(1.)*10**(-7)):
        self.rbm_trainers = []
        self.target_dbn = dbn
        self.train_set = train_set
        self.valid_set = valid_set
        self.image_loader = image_loader
        self.stats_collection_period = stats_collection_period
        self.num_training_epochs = num_training_epochs
        self.learning_rate_eval_period = learning_rate_eval_period
        self.lr_error_multiplier = lr_error_multiplier
        count = 0
        for layer in self.target_dbn.layers:
            trainer = rbm_trainer.RbmTrainer(layer, train_set, valid_set, stats_collection_period, num_training_epochs,
                                             output_directory)

            self.rbm_trainers.append(trainer)
            count += 1

    def train_dbn_unsupervised(self, starting_layer=0, ending_layer=-1):
        if ending_layer == -1:
            ending_layer = len(self.target_dbn.layers) - 1

        for layer_idx in range(starting_layer, ending_layer + 1):
            print "TRAINING LAYER {}".format(layer_idx)
            trainer = self.rbm_trainers[layer_idx]
            recreation_err_squared = []
            for epoch_num in range(0,self.num_training_epochs):
                np.random.shuffle(self.train_set)
                iteration_count = 0
                for image in self.train_set:
                    sample = image.copy()
                    if self.image_loader is not None:
                        sample = self.image_loader.load_image(sample)
                    if layer_idx != 0:
                        sample = self.target_dbn.sample_h_given_v(sample, ending_layer=layer_idx - 1)[1]
                    hidden_bias_delta, sparsity_delta, \
                    visible_bias_delta, weight_group_delta = trainer.train_given_sample(sample)

                    if iteration_count % self.learning_rate_eval_period == 0:
                        layer_recreation = self.target_dbn.layers[layer_idx].gibbs_vhv(sample)[0]
                        layer_recreation_err_sqrd = ((sample - layer_recreation)**2).sum()
                        print ("Layer recreation error = {}".format(layer_recreation_err_sqrd))
                        print ("Setting learning rate to {}".format(self.lr_error_multiplier*layer_recreation_err_sqrd))
                        self.target_dbn.layers[layer_idx].set_learning_rate(self.lr_error_multiplier*layer_recreation_err_sqrd)

                    if (iteration_count % self.stats_collection_period == 0):
                        test_idx = np.random.randint(1, len(self.valid_set) - 1)
                        test_sample = self.valid_set[test_idx].copy()
                        if self.image_loader is not None:
                            test_sample = self.image_loader.load_image(test_sample)
                        recreation = self.target_dbn.recreation_at_layer(test_sample, layer_idx)
                        visualized_filters = self.visualize_filters(layer_idx)

                        trainer.collect_statistics(hidden_bias_delta, iteration_count, recreation_err_squared,
                                                   sparsity_delta, weight_group_delta, epoch_num, test_sample, recreation,
                                                   visualized_filters)



                    iteration_count += 1

    def visualize_filters(self, layer_idx):
        num_bases = self.target_dbn.layers[layer_idx].numBases
        filter_collector = []
        for i in range(0, num_bases):
            hidden_layer_input = np.zeros((1, num_bases, 1, 1))
            hidden_layer_input[0, i, 0, 0] = 1
            outp = self.target_dbn.sample_v_given_h(hidden_layer_input, starting_layer=layer_idx)
            #p2 = np.percentile(outp[0][0][0], 2)
            #p98 = np.percentile(outp[0][0][0], 98)
            #filter_collector.append(exposure.rescale_intensity(outp[0][0][0], in_range=(p2, p98)))
            filter_collector.append(outp[0][0][0])

        allOutp = np.array(filter_collector)
        factors = self.__get_biggest_factors_of(allOutp.shape[0])
        return self.__unblockshaped(allOutp, factors[0] * allOutp.shape[1], factors[1] * allOutp.shape[2])

    def __unblockshaped(self, arr, h, w):
        """
        Return an array of shape (h, w) where
        h * w = arr.size

        If arr is of shape (n, nrows, ncols), n sublocks of shape (nrows, ncols),
        then the returned array preserves the "physical" layout of the sublocks.
        """
        n, nrows, ncols = arr.shape
        return (arr.reshape(h // nrows, -1, nrows, ncols)
                .swapaxes(1, 2)
                .reshape(h, w))

    def __get_biggest_factors_of(self, size):
        factors = list(reduce(list.__add__,
                              ([i, size // i] for i in range(1, int(size ** 0.5) + 1) if size % i == 0)))
        return factors[len(factors) - 2:]
