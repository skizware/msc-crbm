import numpy as np
import matplotlib.pyplot as plt


class RbmTrainer(object):
    def __init__(self, crbm, train_set, valid_set, stats_collection_period=500, num_training_epochs=1,
                 output_directory=''):
        self.targetRbm = crbm
        self.learning_rate = crbm.LEARNING_RATE
        self.target_sparsity = crbm.TARGET_SPARSITY
        self.sparsity_regularization_rate = crbm.REGULARIZATION_RATE
        self.training_set = train_set.copy()
        self.validation_set = valid_set.copy()
        self.stats_collection_period = stats_collection_period
        self.num_epochs = num_training_epochs
        self.output_dir = output_directory

    def train_rbm_unsupervised(self):
        recreation_err_sqrd = []
        weight_updates = []
        hidden_bias_updates = []
        sparsity_updates = []
        visible_bias_updates = []

        for i in range(0, self.num_epochs):
            current_epoch = i + 1
            for test_index in range(0, self.training_set.shape[0]):
                #test_sample = self.__normalize_image(self.training_set[test_index].copy())
                test_sample = self.training_set[test_index].copy()

                weight_group_delta, hidden_bias_delta, \
                sparsity_delta, bias_updates, visible_bias_delta = self.targetRbm.contrastive_divergence(test_sample)

                if test_index % self.stats_collection_period == 0:
                    self.__collect_statistics(hidden_bias_delta, hidden_bias_updates, test_index,
                                              recreation_err_sqrd,
                                              sparsity_delta, sparsity_updates, visible_bias_delta,
                                              visible_bias_updates, weight_group_delta, weight_updates, current_epoch)

    def __collect_statistics(self, hidden_bias_delta, hidden_bias_updates, index, recreation_err_sqrd,
                             sparsity_delta, sparsity_updates, visible_bias_delta, visible_bias_updates,
                             weight_group_delta, weight_updates, current_epoch):
        self.__add_param_update_metrics_to_collectors(hidden_bias_delta, hidden_bias_updates, sparsity_delta,
                                                      sparsity_updates,
                                                      visible_bias_delta, visible_bias_updates, weight_group_delta,
                                                      weight_updates)

        test_idx = np.random.randint(1, len(self.validation_set) - 1)
        #test_sample = self.__normalize_image(self.validation_set[test_idx].copy())
        test_sample = self.validation_set[test_idx].copy()

        recreation, test_sample = self.__get_recreation(test_sample)
        self.__plot_and_save_sample_recreation_comparison(index, recreation, test_sample, current_epoch)
        self.__plot_and_save_recreation_squared_error(recreation, recreation_err_sqrd, test_sample)
        self.__plot_and_save_weight_histograms(index, weight_group_delta, current_epoch)
        self.__plot_and_save_hidden_bias_histograms(hidden_bias_delta, index, sparsity_delta, current_epoch)
        self.__plot_and_save_learned_filters(index, current_epoch)

    def __get_recreation(self, testSample):
        testSample = testSample - testSample.mean()
        testSample = testSample / testSample.std()
        recreation = self.targetRbm.gibbs_vhv(testSample)
        return recreation, testSample

    def __add_param_update_metrics_to_collectors(self, hidden_bias_delta, hidden_bias_updates, sparsity_delta,
                                                 sparsity_updates, visible_bias_delta, visible_bias_updates,
                                                 weight_group_delta, weight_updates):
        weight_updates.append(weight_group_delta.reshape(weight_group_delta.size))
        hidden_bias_updates.append(hidden_bias_delta[0])
        sparsity_updates.append(sparsity_delta)
        visible_bias_updates.append(visible_bias_delta)

    def __plot_and_save_sample_recreation_comparison(self, index, recreation, test_sample, current_epoch):
        fig = plt.figure()
        orig = fig.add_subplot(1, 2, 1)
        orig.set_title('Original')
        plt.imshow(test_sample[0][0], cmap='gray')
        recreat = fig.add_subplot(1, 2, 2)
        recreat.set_title('Recreation')
        plt.imshow(recreation[0][0][0], cmap='gray')
        fig.savefig(
            self.__get_subdir_name() + '/recreations/recreation_' + str(current_epoch) + '_' + str(index) + '.png')
        plt.close(fig)

    def __plot_and_save_recreation_squared_error(self, recreation, recreation_err_sqrd, test_sample):
        recreation_err_sqrd.append(((test_sample - recreation[0]) ** 2).sum())
        fig = plt.figure()
        fig.add_subplot(1, 1, 1).set_title('Recreation squared error')
        plt.plot(recreation_err_sqrd)
        fig.savefig(self.__get_subdir_name() + "/recreation_squared_error.png")
        plt.close(fig)

    def __plot_and_save_weight_histograms(self, index, weight_group_delta, current_epoch):
        fig, axes = plt.subplots(2, 1)
        for axis in axes.flat:
            axis.get_xaxis().set_ticks([])
            axis.get_yaxis().set_ticks([])

        fig_weights = fig.add_subplot(2, 1, 1)
        fig_weights.set_title('Weight Values')
        plt.hist(self.targetRbm.th_weight_groups.get_value().reshape(self.targetRbm.th_weight_groups.get_value().size)
                 , bins=100)
        fig_deltas = fig.add_subplot(2, 1, 2)
        fig_deltas.set_title('Weight Deltas')
        plt.hist(weight_group_delta.reshape(weight_group_delta.size), bins=100)
        fig.tight_layout()
        fig.savefig(self.__get_subdir_name() + '/histograms/histogram_weights_' + str(current_epoch) + '_' + str(
            index) + '.png')
        plt.close(fig)

    def __plot_and_save_hidden_bias_histograms(self, hidden_bias_delta, index, sparsity_delta, current_epoch):
        fig, axes = plt.subplots(3, 1)
        for axis in axes.flat:
            axis.get_xaxis().set_ticks([])
            axis.get_yaxis().set_ticks([])

        fig_hidden_biases = fig.add_subplot(3, 1, 1)
        fig_hidden_biases.set_title("Hidden Biases")
        plt.hist(self.targetRbm.th_hidden_group_biases.get_value(), bins=20)
        fig_hidden_bias_deltas = fig.add_subplot(3, 1, 2)
        fig_hidden_bias_deltas.set_title("Hidden Bias Deltas")
        plt.hist(hidden_bias_delta.reshape(hidden_bias_delta.size), bins=20)
        fig_hidden_bias_deltas = fig.add_subplot(3, 1, 3)
        fig_hidden_bias_deltas.set_title("Sparsity Deltas")
        plt.hist(sparsity_delta.reshape(sparsity_delta.size), bins=20)
        fig.tight_layout()
        fig.savefig(self.__get_subdir_name() + '/histograms/histogram_hidden_biases_' + str(current_epoch) + '_' + str(
            index) + '.png')
        plt.close(fig)

    def __plot_and_save_learned_filters(self, index, current_epoch):
        fig, _ = plt.subplots()
        fig_learned_filters = fig.add_subplot(1, 1, 1)
        fig_learned_filters.set_title("Learned Filters")
        best_shape_arr = self.__get_squarest_shape_arr(self.targetRbm.th_weight_groups.get_value().size)
        fig.tight_layout()
        plt.imshow(self.__unblockshaped(self.targetRbm.th_weight_groups.get_value()[:, 0, :, :], best_shape_arr[1],
                                        best_shape_arr[0]), cmap='gray')
        fig.savefig(self.__get_subdir_name() + '/recreations/filters_' + str(current_epoch) + '_' + str(index) + '.png')
        plt.close(fig)

    def __get_subdir_name(self):
        return self.output_dir + str(self.learning_rate) + '_' + str(self.target_sparsity) + '_' + str(
            self.sparsity_regularization_rate)

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

    def __normalize_image(self, input_image):
        input_image = input_image - input_image.mean()
        input_image = input_image / input_image.std()
        return input_image

    def __get_squarest_shape_arr(self, size):
        factors = list(reduce(list.__add__,
                              ([i, size // i] for i in range(1, int(size ** 0.5) + 1) if size % i == 0)))
        return factors[len(factors) - 2:]
