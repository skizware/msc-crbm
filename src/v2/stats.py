import matplotlib.pyplot as plt
import os
import datetime

DIR_RECREATIONS = 'recreation/'
DIR_HISTOGRAMS = 'histogram/'
FILE_NAME_HID_BIAS_HISTOGRAM = 'sample_{}_vis_bias_hist.png'
FILE_NAME_FILTER_WEIGHT_HISTOGRAM = 'sample_{}_weight_hist.png'
FILE_NAME_RECREATION = 'sample_{}_recreation.png'
FILE_NAME_FILTER_VIS = 'sample_{}_filter_vis.png'


class MultiChannelPlottingDbnTrainingStatsCollector(object):
    def __init__(self, dir_base_output, stats_collection_period=1000):
        self.dir_base_output = dir_base_output
        self.stats_collection_period = stats_collection_period

        if not os.path.isdir(self.dir_base_output + DIR_RECREATIONS):
            os.makedirs(self.dir_base_output + DIR_RECREATIONS)

        if not os.path.isdir(self.dir_base_output + DIR_HISTOGRAMS):
            os.makedirs(self.dir_base_output + DIR_HISTOGRAMS)

    def collect_stats(self, weight_group_delta, hidden_bias_delta, sparsity_delta, bias_updates, visible_bias_delta,
                      pos_hid_infer, neg_vis_infer, neg_hid_infer, original_input, sample_number, dbn):
        if sample_number % self.stats_collection_period is 0:
            print('Collecting stats for sample {} at {}'.format(sample_number, datetime.datetime.now()))
            trained_layer_idx = len(dbn.layers) - 1
            hidden_value = dbn.layers[trained_layer_idx].get_visible_units().get_value() if trained_layer_idx is not 0 else dbn.layers[trained_layer_idx].get_hidden_units().get_value()
            network_recreation = dbn.infer_vis_given_hid(hidden_value,
                                                         start_layer_index=trained_layer_idx - 1, end_layer_index_incl=-1)

            self.__plot_and_save_sample_recreation_comparison(sample_number, network_recreation, original_input, 1)

    def __plot_and_save_sample_recreation_comparison(self, index, recreation, test_sample, current_epoch):
        fig = plt.figure()
        orig = fig.add_subplot(1, 2, 1)
        orig.set_title('Original')
        plt.imshow(test_sample[0][0], cmap='gray')
        recreat = fig.add_subplot(1, 2, 2)
        recreat.set_title('Recreation')
        plt.imshow(recreation[0][0][0], cmap='gray')
        fig.savefig(
            self.dir_base_output + DIR_RECREATIONS + FILE_NAME_RECREATION.format(index))
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
