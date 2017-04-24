from data import MnistDataLoader, NormalizingCropOrPadToSizeImageLoader
import dbn
from train import DbnTrainer
from abc import ABCMeta, abstractmethod
import copy
import cPickle
import gzip
import numpy as np
from dbn import BinaryVisibleDbn, GaussianVisibleDbn
from stats import MultiChannelPlottingDbnTrainingStatsCollector
import os
import traceback

MNIST_DATA_SET_PATH = '/home/dave/data/mnist.pkl.gz'
CALTEC_DATA_SET_PATH = '/home/dave/data/101_ObjectCategories/'
KEY_VIS_SHAPE = 'vis_shape'
KEY_HID_SHAPE = 'hid_shape'
KEY_POOL_RATIO = 'pooling_ratio'
KEY_LEARNING_RATES = 'learning_rates'
KEY_TARGET_SPARSITIES = 'target_sparsities'
KEY_SPARSITY_LEARNING_RATES = 'sparsity_learrning_rates'
KEY_LAYER_TYPE = 'layer_type'
DIR_OUT_RESULTS = 'mnist_pooled/results/'


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
                                       target_sparsity=target_sparsity, sparsity_learning_rate=sparsity_learning_rate, pooling_ratio=grids[KEY_POOL_RATIO])

                    trainer = DbnTrainer(dbn_copy, self.train_set, self.get_data_loader(),
                                         self.get_stats_collector(self.get_dbn_output_dir(dbn_copy)),
                                         self.get_dbn_output_dir(dbn_copy))
                    try:
                        trainer.train_dbn(len(dbn_copy.layers) - 1)
                        trainer.save_state()
                        results["{}_{}_{}".format(learning_rate, target_sparsity, sparsity_learning_rate)] = dbn_copy
                    except Exception, e:
                        print e
                        with open(self.get_dbn_output_dir(dbn_copy) + 'error.txt', 'w') as f:
                            traceback.print_exc(file=f)
                        trainer.save_state("AT_ERROR")

        return results

    @staticmethod
    def load_data_sets():
        pass

    @staticmethod
    def get_data_loader(self):
        pass

    @abstractmethod
    def get_stats_collector(self, results_output_dir):
        pass

    def get_dbn_output_dir(self, dbn):
        out = DIR_OUT_RESULTS
        for layer_idx in xrange(0, len(dbn.layers)):
            dbn_layer = dbn.layers[layer_idx]
            out += "layer_{}/lr_{}_st_{}_slr_{}/".format(layer_idx, dbn_layer.get_learning_rate(),
                                                         dbn_layer.get_target_sparsity(),
                                                         dbn_layer.get_sparsity_learning_rate())

        return out


class MnistExperiment(AbstractDbnGridSearchExperiment):
    def __init__(self, pre_initialized_dbn):
        super(MnistExperiment, self).__init__(pre_initialized_dbn)

    @staticmethod
    def get_data_loader():
        return MnistDataLoader()

    @staticmethod
    def load_data_sets():
        f = gzip.open(MNIST_DATA_SET_PATH, "rb")
        train_set, valid_set, test_set = cPickle.load(f)
        f.close()
        return train_set[0], valid_set[0], test_set[0]

    def get_stats_collector(self, results_output_dir):
        return MultiChannelPlottingDbnTrainingStatsCollector(results_output_dir)


class CaltechExperiment(AbstractDbnGridSearchExperiment):
    def __init__(self, pre_initialized_dbn):
        super(CaltechExperiment, self).__init__(pre_initialized_dbn)

    @staticmethod
    def get_data_loader():
        return NormalizingCropOrPadToSizeImageLoader(300, 200, grayscale=True)

    @staticmethod
    def load_data_sets():
        with os.popen('find {} -name *.jpg'.format(CALTEC_DATA_SET_PATH)) as f:
            image_refs_unsupervised = f.read().split('\n')

        train_set = image_refs_unsupervised[:len(image_refs_unsupervised) - 1]
        return train_set, None, None

    def get_stats_collector(self, results_output_dir):
        return MultiChannelPlottingDbnTrainingStatsCollector(results_output_dir)

# starting_dbn = dbn.DbnFromStateBuilder.init_dbn(np.load('/home/dave/code/msc-crbm/test/v2/caltech/results/layer_0/lr_0.001_st_1.0_slr_0.0/dbn_state.npy').item())
starting_dbn = dbn.BinaryVisibleDbn()
mnistExp = MnistExperiment(starting_dbn)
# caltechExp = CaltechExperiment(starting_dbn)

grids_example = {
    KEY_VIS_SHAPE: (1, 1, 28, 28),
    KEY_HID_SHAPE: (1, 32, 20, 20),
    KEY_POOL_RATIO: 2,
    KEY_LEARNING_RATES: [0.01],
    KEY_TARGET_SPARSITIES: [0.1],
    KEY_SPARSITY_LEARNING_RATES: [0.1]
}

resultant2 = mnistExp.run_grids(grids_example)
