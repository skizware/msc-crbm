import numpy as np
import matplotlib.pyplot as plt
import crbm
import os


class RbmTrainer(object):
    def __init__(self, crbm, train_set, valid_set, stats_collection_period=500, num_training_epochs=1,
                 output_directory='./', image_loader=None):
        self.targetRbm = crbm
        self.learning_rate = crbm.LEARNING_RATE
        self.target_sparsity = crbm.TARGET_SPARSITY
        self.sparsity_regularization_rate = crbm.REGULARIZATION_RATE
        self.training_set = train_set.copy()
        self.validation_set = valid_set.copy()
        self.stats_collection_period = stats_collection_period
        self.num_epochs = num_training_epochs
        self.output_dir = output_directory
        self.image_loader = image_loader
        if not os.path.exists(self.output_dir + 'recreations/'):
            os.mkdir(self.output_dir + 'recreations/')
        if not os.path.exists(self.output_dir + 'histograms/'):
            os.mkdir(self.output_dir + 'histograms/')

    def train_rbm_unsupervised(self):
        recreation_err_sqrd = []

        for i in range(0, self.num_epochs):
            current_epoch = i + 1
            test_index = 0
            for image_ref in self.training_set:
                hidden_bias_delta, sparsity_delta, \
                visible_bias_delta, weight_group_delta = self.train_given_sample(image_ref)

                if test_index % self.stats_collection_period == 0:
                    test_idx = np.random.randint(1, len(self.validation_set) - 1)
                    test_sample = self.validation_set[test_idx].copy()
                    if type(self.targetRbm) == crbm.crbm:
                        test_sample = self.normalize_image(test_sample)

                    recreation = self.targetRbm.gibbs_vhv(test_sample)
                    self.collect_statistics(hidden_bias_delta, test_index, recreation_err_sqrd, sparsity_delta,
                                            weight_group_delta, current_epoch, test_sample, recreation)

                test_index += 1

    def train_given_sample(self, image_ref, previous_layer, previous_layer_visible_sample):
        if self.image_loader is not None:
            test_sample = self.image_loader.load_image(image_ref).copy()
        else:
            test_sample = image_ref.copy()
        weight_group_delta, hidden_bias_delta, \
        sparsity_delta, bias_updates, visible_bias_delta = self.targetRbm.contrastive_divergence(test_sample,
                                                                                                 previous_layer,
                                                                                                 previous_layer_visible_sample)
        return hidden_bias_delta, sparsity_delta, visible_bias_delta, weight_group_delta

    def collect_statistics(self, hidden_bias_delta, index, recreation_err_sqrd, sparsity_delta, weight_group_delta,
                           current_epoch, test_sample, recreation, visualized_filters):

        print("Collecting stats for sample {}".format(index))
        self.__plot_and_save_sample_recreation_comparison(index, recreation, test_sample, current_epoch)
        self.__plot_and_save_recreation_squared_error(recreation, recreation_err_sqrd, test_sample)
        self.__plot_and_save_weight_histograms(index, weight_group_delta, current_epoch)
        self.__plot_and_save_hidden_bias_histograms(hidden_bias_delta, index, sparsity_delta, current_epoch)
        self.__plot_and_save_learned_filters(index, current_epoch, visualized_filters)

    def __plot_and_save_sample_recreation_comparison(self, index, recreation, test_sample, current_epoch):
        fig = plt.figure()
        orig = fig.add_subplot(1, 2, 1)
        orig.set_title('Original')
        plt.imshow(test_sample[0][0], cmap='gray')
        recreat = fig.add_subplot(1, 2, 2)
        recreat.set_title('Recreation')
        plt.imshow(recreation[0][0][0], cmap='gray')
        fig.savefig(
            self.__get_subdir_name() + 'recreations/recreation_' + str(current_epoch) + '_' + str(index) + '.png')
        plt.close(fig)

    def __plot_and_save_recreation_squared_error(self, recreation, recreation_err_sqrd, test_sample):
        recreation_err_sqrd.append(((test_sample - recreation[0]) ** 2).sum())
        fig = plt.figure()
        fig.add_subplot(1, 1, 1).set_title('Recreation squared error')
        plt.plot(recreation_err_sqrd)
        fig.savefig(self.__get_subdir_name() + "recreation_squared_error.png")
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
        fig.savefig(self.__get_subdir_name() + 'histograms/histogram_weights_' + str(current_epoch) + '_' + str(
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
        fig.savefig(self.__get_subdir_name() + 'histograms/histogram_hidden_biases_' + str(current_epoch) + '_' + str(
            index) + '.png')
        plt.close(fig)

    def __plot_and_save_learned_filters(self, index, current_epoch, visualized_filters, cmap_val='gray'):
        fig, _ = plt.subplots()
        fig_learned_filters = fig.add_subplot(1, 1, 1)
        fig_learned_filters.set_title("Learned Filters")

        plt.imshow(visualized_filters, cmap=cmap_val)
        fig.tight_layout()
        fig.savefig(self.__get_subdir_name() + 'recreations/filters_' + str(current_epoch) + '_' + str(index) + '.png')
        plt.close(fig)

    def __get_subdir_name(self):
        return self.output_dir

    def normalize_image(self, input_image):
        input_image = input_image - input_image.mean()
        input_image = input_image / input_image.std()
        return input_image
