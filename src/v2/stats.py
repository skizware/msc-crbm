import matplotlib.pyplot as plt
import os
import datetime
import numpy as np

DIR_RECREATIONS = 'recreation/'
DIR_HISTOGRAMS = 'histogram/'
FILE_NAME_HID_BIAS_HISTOGRAM = 'hidden_bias_hist_sample_{}.png'
FILE_NAME_FILTER_WEIGHT_HISTOGRAM = 'weight_hist_sample_{}.png'
FILE_NAME_RECREATION = 'sample_{}_recreation.png'
FILE_NAME_FILTER_VIS = 'sample_{}_filter_vis.png'
FILE_NAME_CHAIN_SAMPLE = 'sample_{}_from_chain.png'


class MultiChannelPlottingDbnTrainingStatsCollector(object):
    def __init__(self, dir_base_output, stats_collection_period=1):
        self.dir_base_output = dir_base_output
        self.stats_collection_period = stats_collection_period
        self.recreation_error_sqrd_collector = []

        if not os.path.isdir(self.dir_base_output + DIR_RECREATIONS):
            os.makedirs(self.dir_base_output + DIR_RECREATIONS)

        if not os.path.isdir(self.dir_base_output + DIR_HISTOGRAMS):
            os.makedirs(self.dir_base_output + DIR_HISTOGRAMS)

    def collect_stats(self, weight_group_delta, hidden_bias_delta, sparsity_delta, bias_updates, visible_bias_delta,
                      pos_hid_infer, neg_vis_infer, neg_hid_infer, layer_rec_err_sqrd, original_input, sample_number,
                      dbn, neg_vis_sampled):
        if sample_number % self.stats_collection_period is 0:
            print('Collecting stats for sample {} at {}'.format(sample_number, datetime.datetime.now()))
            trained_layer_idx = len(dbn.layers) - 1
            network_recreation = dbn.infer_vis_given_hid(neg_vis_sampled, start_layer_index=trained_layer_idx - 1,
                                                         end_layer_index_incl=-1)[
                0] if trained_layer_idx is not 0 else neg_vis_infer

            self.__plot_and_save_recreation_squared_error(layer_rec_err_sqrd, self.recreation_error_sqrd_collector)
            self.__plot_and_save_hidden_bias_histograms(hidden_bias_delta, sample_number, sparsity_delta, 1, dbn)
            self.__plot_and_save_weight_histograms(sample_number, weight_group_delta, 1, dbn)
            self.__plot_and_save_sample_recreation_comparison(sample_number, network_recreation, original_input, 1)
            #TODO - fix filter visualization for pooled layers
            filters_by_channel, unblock_shape = self.visualize_filters(trained_layer_idx, dbn)
            self.__plot_and_save_learned_filters(sample_number, filters_by_channel, unblock_shape)

    def __plot_and_save_learned_filters(self, index, filters_by_channel, unblock_shape, cmap_val='gray'):
        num_channels = filters_by_channel.shape[1]
        fig = plt.figure()
        for i in xrange(0, num_channels):
            filters_for_channel = filters_by_channel[:, i, :, :]
            filters = unblockshaped(filters_for_channel, unblock_shape[0], unblock_shape[1])
            filters_fig = fig.add_subplot(num_channels, 1, i+1)
            filters_fig.set_title("Filters - Channel {}".format(i))
            plt.imshow(filters, cmap='gray')

        fig.tight_layout()
        fig.savefig(
            self.dir_base_output + DIR_RECREATIONS + FILE_NAME_FILTER_VIS.format(index))
        plt.close(fig)

    def __plot_and_save_sample_recreation_comparison(self, index, recreation, test_sample, current_epoch):
        num_channels = test_sample.shape[1]
        fig = plt.figure()

        batch_factors = get_biggest_factors_of(test_sample.shape[0])
        unblock_w = batch_factors[0] * test_sample.shape[2]
        unblock_h = batch_factors[1] * test_sample.shape[3]

        for i in range(0, num_channels):
            orig = fig.add_subplot(num_channels, 2, i + 1)
            orig.set_title('Original - Channel {}'.format(i))
            plt.imshow(unblockshaped(test_sample[:, i, :, :], unblock_w, unblock_h), cmap='gray')

            recreat = fig.add_subplot(num_channels, 2, i + 2)
            recreat.set_title("Recreation - Channel {}".format(i))
            plt.imshow(unblockshaped(recreation[:, i, :, :], unblock_w, unblock_h), cmap='gray')

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

    def visualize_filters(self, layer_idx, dbn):
        num_bases = dbn.layers[layer_idx].get_hidden_units().get_shape()[1]
        filter_collector = []
        for i in xrange(0, num_bases):
            hidden_layer_input = np.zeros((1, num_bases, 1, 1))
            hidden_layer_input[0, i, 0, 0] = 1
            outp = dbn.infer_vis_given_hid(hidden_layer_input, start_layer_index=layer_idx)
            # p2 = np.percentile(outp[0][0][0], 2)
            # p98 = np.percentile(outp[0][0][0], 98)
            # filter_collector.append(exposure.rescale_intensity(outp[0][0][0], in_range=(p2, p98)))
            filter_collector.append(outp[0])

        allOutp = np.array(filter_collector)
        factors = get_biggest_factors_of(allOutp.shape[0])
        collected_output = allOutp[:, 0, :, :, :]
        return collected_output, (factors[0] * collected_output.shape[2], factors[1] * collected_output.shape[3])
        #return unblockshaped(allOutp, factors[0] * allOutp.shape[1], factors[1] * allOutp.shape[2])


class MultiChannelPlottingPersistentChainDbnTrainingStatsCollector(MultiChannelPlottingDbnTrainingStatsCollector):
    def collect_stats(self, weight_group_delta, hidden_bias_delta, sparsity_delta, bias_updates, visible_bias_delta,
                      pos_hid_infer, neg_vis_infer, neg_hid_infer, layer_rec_err_sqrd, original_input, sample_number,
                      dbn, neg_vis_sampled):

        if sample_number % self.stats_collection_period is 0:
            self.__plot_and_save_chain_sample(sample_number, neg_vis_infer, 1)

        pos_hid_sample = dbn.infer_hid_given_vis(original_input)[1]
        neg_vis_infer = dbn.infer_vis_given_hid(pos_hid_sample)[0]

        layer_rec_err_sqrd = ((original_input - neg_vis_infer) ** 2).sum() / original_input.shape[0]

        super(MultiChannelPlottingPersistentChainDbnTrainingStatsCollector, self).collect_stats(weight_group_delta,
                                                                                                hidden_bias_delta,
                                                                                                sparsity_delta,
                                                                                                bias_updates,
                                                                                                visible_bias_delta,
                                                                                                pos_hid_infer,
                                                                                                neg_vis_infer,
                                                                                                neg_hid_infer,
                                                                                                layer_rec_err_sqrd,
                                                                                                original_input,
                                                                                                sample_number, dbn,
                                                                                                neg_vis_sampled)

    def __plot_and_save_chain_sample(self, index, chain_sample, current_epoch):
        num_channels = chain_sample.shape[1]
        fig = plt.figure()

        batch_factors = get_biggest_factors_of(chain_sample.shape[0])
        unblock_w = batch_factors[0] * chain_sample.shape[2]
        unblock_h = batch_factors[1] * chain_sample.shape[3]

        for i in range(0, num_channels):
            orig = fig.add_subplot(num_channels, 1, i + 1)
            orig.set_title('Chain Sample - Channel {}'.format(i))
            plt.imshow(unblockshaped(chain_sample[:,i,:,:], unblock_w, unblock_h), cmap='gray')

        fig.tight_layout()
        fig.savefig(
            self.dir_base_output + DIR_RECREATIONS + FILE_NAME_CHAIN_SAMPLE.format(index))
        plt.close(fig)


def get_biggest_factors_of(size):
    factors = list(reduce(list.__add__,
                          ([i, size // i] for i in range(1, int(size ** 0.5) + 1) if size % i == 0)))
    return factors[len(factors) - 2:]

def unblockshaped(arr, h, w):
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
