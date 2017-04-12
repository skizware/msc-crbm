import matplotlib.pyplot as plt
import os
import datetime

DIR_RECREATIONS = 'recreation/'
DIR_HISTOGRAMS = 'histogram/'
FILE_NAME_HID_BIAS_HISTOGRAM = 'hidden_bias_hist_sample_{}.png'
FILE_NAME_FILTER_WEIGHT_HISTOGRAM = 'weight_hist_sample_{}.png'
FILE_NAME_RECREATION = 'sample_{}_recreation.png'
FILE_NAME_FILTER_VIS = 'sample_{}_filter_vis.png'


class MultiChannelPlottingDbnTrainingStatsCollector(object):
    def __init__(self, dir_base_output, stats_collection_period=1000):
        self.dir_base_output = dir_base_output
        self.stats_collection_period = stats_collection_period
        self.recreation_error_sqrd_collector = []

        if not os.path.isdir(self.dir_base_output + DIR_RECREATIONS):
            os.makedirs(self.dir_base_output + DIR_RECREATIONS)

        if not os.path.isdir(self.dir_base_output + DIR_HISTOGRAMS):
            os.makedirs(self.dir_base_output + DIR_HISTOGRAMS)

    def collect_stats(self, weight_group_delta, hidden_bias_delta, sparsity_delta, bias_updates, visible_bias_delta,
                      pos_hid_infer, neg_vis_infer, neg_hid_infer, layer_rec_err_sqrd, original_input, sample_number,
                      dbn):
        if sample_number % self.stats_collection_period is 0:
            print('Collecting stats for sample {} at {}'.format(sample_number, datetime.datetime.now()))
            trained_layer_idx = len(dbn.layers) - 1
            network_recreation = dbn.infer_vis_given_hid(neg_vis_infer, start_layer_index=trained_layer_idx - 1,
                                                         end_layer_index_incl=-1)[
                0] if trained_layer_idx is not 0 else neg_vis_infer

            self.__plot_and_save_recreation_squared_error(layer_rec_err_sqrd, self.recreation_error_sqrd_collector)
            self.__plot_and_save_hidden_bias_histograms(hidden_bias_delta, sample_number, sparsity_delta, 1, dbn)
            self.__plot_and_save_weight_histograms(sample_number, weight_group_delta, 1, dbn)
            self.__plot_and_save_sample_recreation_comparison(sample_number, network_recreation, original_input, 1)

    def __plot_and_save_sample_recreation_comparison(self, index, recreation, test_sample, current_epoch):
        num_channels = test_sample.shape[1]
        fig = plt.figure()

        for i in range(0, num_channels):
            orig = fig.add_subplot(num_channels, 2, i + 1)
            orig.set_title('Original - Channel {}'.format(i))
            plt.imshow(test_sample[0][i], cmap='gray')

            recreat = fig.add_subplot(num_channels, 2, i + 2)
            recreat.set_title("Recreation - Channel {}".format(i))
            plt.imshow(recreation[0][i], cmap='gray')

        fig.tight_layout()
        fig.savefig(
            self.dir_base_output + DIR_RECREATIONS + FILE_NAME_RECREATION.format(index))
        plt.close(fig)

    def __plot_and_save_recreation_squared_error(self, recreation_err_sqrd, collector):
        collector.append(recreation_err_sqrd)
        fig = plt.figure()
        fig.add_subplot(1, 1, 1).set_title('Recreation squared error')
        plt.plot(collector)
        fig.tight_layout()
        fig.savefig(self.dir_base_output + "recreation_squared_error.png")
        plt.close(fig)

    def __plot_and_save_hidden_bias_histograms(self, hidden_bias_delta, index, sparsity_delta, current_epoch, dbn):
        fig, axes = plt.subplots(3, 1)
        for axis in axes.flat:
            axis.get_xaxis().set_ticks([])
            axis.get_yaxis().set_ticks([])

        fig_hidden_biases = fig.add_subplot(3, 1, 1)
        fig_hidden_biases.set_title("Hidden Biases")
        trained_layer_hidden_biases = dbn.layers[-1].get_hidden_biases().get_value()
        plt.hist(trained_layer_hidden_biases.reshape(trained_layer_hidden_biases.size), bins=20)
        fig_hidden_bias_deltas = fig.add_subplot(3, 1, 2)
        fig_hidden_bias_deltas.set_title("Hidden Bias Deltas")
        plt.hist(hidden_bias_delta.reshape(hidden_bias_delta.size), bins=20)
        fig_hidden_bias_deltas = fig.add_subplot(3, 1, 3)
        fig_hidden_bias_deltas.set_title("Sparsity Deltas")
        plt.hist(sparsity_delta.reshape(sparsity_delta.size), bins=20)
        fig.tight_layout()
        fig.savefig(self.dir_base_output + DIR_HISTOGRAMS + FILE_NAME_HID_BIAS_HISTOGRAM.format(index))
        plt.close(fig)

    def __plot_and_save_weight_histograms(self, index, weight_group_delta, current_epoch, dbn):
        fig, axes = plt.subplots(2, 1)
        for axis in axes.flat:
            axis.get_xaxis().set_ticks([])
            axis.get_yaxis().set_ticks([])

        fig_weights = fig.add_subplot(2, 1, 1)
        fig_weights.set_title('Weight Values')
        plt.hist(dbn.layers[-1].get_weight_matrix().get_value().reshape(dbn.layers[-1].get_weight_matrix().get_value().size)
                 , bins=100)
        fig_deltas = fig.add_subplot(2, 1, 2)
        fig_deltas.set_title('Weight Deltas')
        plt.hist(weight_group_delta.reshape(weight_group_delta.size), bins=100)
        fig.tight_layout()
        fig.savefig(self.dir_base_output + DIR_HISTOGRAMS + FILE_NAME_FILTER_WEIGHT_HISTOGRAM.format(index))
        plt.close(fig)


def unblockshaped(arr, h, w):
    """
    Return an array of shape (h, w) where
    h * w = arr.size

    If arr is of shape (n, nrows, ncols), n sublocks of shape (nrows, ncols),
    then the returned array preserves the "physical" layout of the sublocks.
    """
    n, nrows, ncols = a.shape
    return (arr.reshape(h // nrows, -1, nrows, ncols)
            .swapaxes(1, 2)
            .reshape(h, w))
