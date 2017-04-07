from data import MnistDataLoader
from train import DbnTrainer
from abc import ABCMeta, abstractmethod
import copy
import cPickle
import gzip
from dbn import BinaryVisibleNonPooledDbn
from stats import MultiChannelPlottingDbnTrainingStatsCollector

MNIST_DATA_SET_PATH = '/home/dave/data/mnist.pkl.gz'
KEY_VIS_SHAPE = 'vis_shape'
KEY_HID_SHAPE = 'hid_shape'
KEY_LEARNING_RATES = 'learning_rates'
KEY_TARGET_SPARSITIES = 'target_sparsities'
KEY_SPARSITY_LEARNING_RATES = 'sparsity_learrning_rates'
KEY_LAYER_TYPE = 'layer_type'


class AbstractDbnGridSearchExperiment(object):
    __metaclass__ = ABCMeta

    def __init__(self, pre_initialized_dbn):
        self.target_dbn = pre_initialized_dbn
        self.train_set, self.valid_set, self.test_set = self.load_data_sets()

    def run_grids(self, grids):
        results = {}
        for sparsity_learning_rate in grids[KEY_SPARSITY_LEARNING_RATES]:
            for target_sparsity in grids[KEY_TARGET_SPARSITIES]:
                for learning_rate in grids[KEY_LEARNING_RATES]:
                    dbn_copy = copy.deepcopy(self.target_dbn)
                    dbn_copy.add_layer(grids[KEY_VIS_SHAPE], grids[KEY_HID_SHAPE], learning_rate=learning_rate,
                                       target_sparsity=target_sparsity, sparsity_learning_rate=sparsity_learning_rate)

                    trainer = DbnTrainer(dbn_copy, self.train_set, self.get_data_loader(), self.get_stats_collector("results/test/"))
                    trainer.train_dbn(len(dbn_copy.layers) - 1)

                    results["{}_{}_{}".format(learning_rate, target_sparsity, sparsity_learning_rate)] = dbn_copy

        return results

    @abstractmethod
    def load_data_sets(self):
        pass

    @abstractmethod
    def get_data_loader(self):
        pass

    @abstractmethod
    def get_stats_collector(self, results_output_dir):
        pass


class MnistExperiment(AbstractDbnGridSearchExperiment):
    def __init__(self, pre_initialized_dbn):
        super(MnistExperiment, self).__init__(pre_initialized_dbn)

    def get_data_loader(self):
        return MnistDataLoader()

    def load_data_sets(self):
        f = gzip.open(MNIST_DATA_SET_PATH, "rb")
        train_set, valid_set, test_set = cPickle.load(f)
        f.close()
        return train_set[0], valid_set[0], test_set[0]

    def get_stats_collector(self, results_output_dir):
        return MultiChannelPlottingDbnTrainingStatsCollector(results_output_dir)


starting_dbn = BinaryVisibleNonPooledDbn()

mnistExp = MnistExperiment(starting_dbn)

grids_example = {
    KEY_VIS_SHAPE: (1, 1, 28, 28),
    KEY_HID_SHAPE: (1, 40, 19, 19),
    KEY_LEARNING_RATES: [0.1],
    KEY_TARGET_SPARSITIES: [0.1],
    KEY_SPARSITY_LEARNING_RATES: [1]
}

mnistExp.run_grids(grids_example)