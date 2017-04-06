from dbn import AbstractDbn
from layer import BinaryVisibleNonPooledLayer
from data import MnistDataLoader
import copy

MNIST_DATA_SET_PATH = '~/data/mnist.pkl.gz'
KEY_VIS_SHAPE = 'vis_shape'
KEY_HID_SHAPE = 'hid_shape'
KEY_LEARNING_RATES = 'learning_rates'
KEY_TARGET_SPARSITIES = 'target_sparsities'
KEY_SPARSITY_LERANING_RATES = 'sparsity_learrning_rates'
KEY_LAYER_TYPE = 'layer_type'


class AbstractDbnGridSearchExperiment(object):
    def run_grids(self, dbn, grids, data):
        for sparsity_learning_rate in grids[KEY_SPARSITY_LERANING_RATES]:
            for target_sparsity in grids[KEY_TARGET_SPARSITIES]:
                for learning_rate in grids[KEY_LEARNING_RATES]:
                    dbn_copy = copy.deepcopy(dbn)
                    dbn_copy.add_layer(grids_example[KEY_HID_SHAPE], learning_rate=learning_rate, target_sparsity=target_sparsity, sparsity_learning_rate=sparsity_learning_rate)
                    train_layer(dbn_copy, data)

    def train_layer(self, dbn, data):
        raise NotImplementedError("Not Yet Implemented")

"""f = gzip.open("/home/dave/Downloads/mnist.pkl.gz", "rb")
train_set, valid_set, test_set = cPickle.load(f)
f.close()"""

"""
grids_example = {
    KEY_LAYER_TYPE: type(BinaryVisibleNonPooledLayer),
    KEY_VIS_SHAPE: (1, 1, 28, 28),
    KEY_HID_SHAPE: (1, 40, 19, 19),
    KEY_LEARNING_RATES: [0.1, 0.01, 0.001],
    KEY_TARGET_SPARSITIES: [0.1, 0.01, 0.001],
    KEY_SPARSITY_LERANING_RATES: [1, 0.1]
}"""
