from data import MnistDataLoader, NormalizingCropOrPadToSizeImageLoader, NumpyArrayStftMagnitudeDataLoader
import dbn
from train import DbnTrainer
from abc import ABCMeta, abstractmethod
import copy
import cPickle
import gzip
import numpy as np
from stats import MultiChannelPlottingDbnTrainingStatsCollector, \
    MultiChannelPlottingPersistentChainDbnTrainingStatsCollector, \
    MultiChannelPlottingDbnTrainingPCAReconstructingStatsCollector
from subprocess import Popen, PIPE
import os
import traceback

MNIST_DATA_SET_PATH = '/home/dave/data/mnist.pkl.gz'
CALTEC_DATA_SET_PATH = '/home/dave/data/101_ObjectCategories/'
TIMIT_DATA_SET_PATH = '/home/dave/code/msc-crbm/test/v2/TIMIT/pre-processed'
KEY_VIS_SHAPE = 'vis_shape'
KEY_HID_SHAPE = 'hid_shape'
KEY_POOL_RATIO = 'pooling_ratio'
KEY_LEARNING_RATES = 'learning_rates'
KEY_TARGET_SPARSITIES = 'target_sparsities'
KEY_SPARSITY_LEARNING_RATES = 'sparsity_learrning_rates'
KEY_LAYER_TYPE = 'layer_type'
KEY_NUM_EPOCHS = 'num_epochs'
KEY_BATCH_SIZE = 'batch_size'
KEY_RE_TRAIN_TOP_LAYER = 'retrain_top_layer'

DIR_OUT_RESULTS = 'mnist_pooled/results/'


class AbstractDbnGridSearchExperiment(object):
    __metaclass__ = ABCMeta

    def __init__(self, pre_initialized_dbn, result_output_dir):
        self.target_dbn = pre_initialized_dbn
        self.train_set, self.valid_set, self.test_set = self.load_data_sets()
        self.result_output_dir = result_output_dir

    def run_grids(self, grids):
        results = {}
        for learning_rate in grids[KEY_LEARNING_RATES]:
            for target_sparsity in grids[KEY_TARGET_SPARSITIES]:
                for sparsity_learning_rate in grids[KEY_SPARSITY_LEARNING_RATES]:
                    dbn_copy = copy.deepcopy(self.target_dbn)
                    if not grids[KEY_RE_TRAIN_TOP_LAYER]:
                        dbn_copy.add_layer(grids[KEY_VIS_SHAPE], grids[KEY_HID_SHAPE], learning_rate=learning_rate,
                                           target_sparsity=target_sparsity, sparsity_learning_rate=sparsity_learning_rate,
                                           pooling_ratio=grids[KEY_POOL_RATIO])
                    else:
                        dbn_copy.layers[-1].set_learning_rate(learning_rate)
                        dbn_copy.layers[-1].set_target_sparsity(target_sparsity)
                        dbn_copy.layers[-1].set_sparsity_learning_rate(sparsity_learning_rate)

                    trainer = DbnTrainer(dbn_copy, self.train_set, self.get_data_loader(),
                                         self.get_stats_collector(self.get_dbn_output_dir(dbn_copy)),
                                         self.get_dbn_output_dir(dbn_copy))
                    try:
                        trainer.train_dbn(len(dbn_copy.layers) - 1, grids[KEY_NUM_EPOCHS], grids[KEY_BATCH_SIZE])
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

    def get_data_loader(self):
        pass

    @abstractmethod
    def get_stats_collector(self, results_output_dir):
        pass

    def get_dbn_output_dir(self, dbn):
        out = self.result_output_dir
        for layer_idx in xrange(0, len(dbn.layers)):
            dbn_layer = dbn.layers[layer_idx]
            out += "layer_{}/lr_{}_st_{}_slr_{}/".format(layer_idx, dbn_layer.get_learning_rate(),
                                                         dbn_layer.get_target_sparsity(),
                                                         dbn_layer.get_sparsity_learning_rate())

        return out


class MnistExperiment(AbstractDbnGridSearchExperiment):
    def __init__(self, pre_initialized_dbn, result_output_dir):
        super(MnistExperiment, self).__init__(pre_initialized_dbn, result_output_dir)

    def get_data_loader(self):
        return MnistDataLoader()

    @staticmethod
    def load_data_sets():
        f = gzip.open(MNIST_DATA_SET_PATH, "rb")
        train_set, valid_set, test_set = cPickle.load(f)
        f.close()
        return train_set[0], valid_set[0], test_set[0]

    @staticmethod
    def load_data_sets_and_labels():
        f = gzip.open(MNIST_DATA_SET_PATH, "rb")
        train_set, valid_set, test_set = cPickle.load(f)
        f.close()
        return train_set, valid_set, test_set

    def get_stats_collector(self, results_output_dir):
        return MultiChannelPlottingDbnTrainingStatsCollector(results_output_dir, 100)


class MnistExperimentPersistentGibbs(MnistExperiment):
    def __init__(self, pre_initialized_dbn, result_output_dir):
        super(MnistExperimentPersistentGibbs, self).__init__(pre_initialized_dbn, result_output_dir)

    def get_stats_collector(self, results_output_dir):
        return MultiChannelPlottingPersistentChainDbnTrainingStatsCollector(results_output_dir, 780)


class CaltechExperiment(AbstractDbnGridSearchExperiment):
    def __init__(self, pre_initialized_dbn, result_output_dir):
        super(CaltechExperiment, self).__init__(pre_initialized_dbn, result_output_dir)

    def get_data_loader(self):
        return NormalizingCropOrPadToSizeImageLoader(300, 200, grayscale=True)

    @staticmethod
    def load_data_sets():
        with os.popen('find {} -name *.jpg'.format(CALTEC_DATA_SET_PATH)) as f:
            image_refs_unsupervised = f.read().split('\n')

        train_set = image_refs_unsupervised[:len(image_refs_unsupervised) - 1]
        return train_set, None, None

    def get_stats_collector(self, results_output_dir):
        return MultiChannelPlottingDbnTrainingStatsCollector(results_output_dir)


class CaltechExperimentPersistentGibbs(CaltechExperiment):
    def __init__(self, pre_initialized_dbn, result_output_dir):
        super(CaltechExperimentPersistentGibbs, self).__init__(pre_initialized_dbn, result_output_dir)

    def get_stats_collector(self, results_output_dir):
        return MultiChannelPlottingPersistentChainDbnTrainingStatsCollector(results_output_dir)


class TimitExperiment(AbstractDbnGridSearchExperiment):
    def __init__(self, pre_initialized_dbn, result_output_dir, pca_model=None, scale_features=None):
        super(TimitExperiment, self).__init__(pre_initialized_dbn, result_output_dir)
        self.pca_model = pca_model
        self.scale_features = scale_features

    def get_data_loader(self):
        return NumpyArrayStftMagnitudeDataLoader(pca_model=self.pca_model, scale_features=self.scale_features)

    @staticmethod
    def load_data_sets():
        proc = Popen(['find', TIMIT_DATA_SET_PATH, '-iname', '*.npy'], stdout=PIPE)
        fileRefs = proc.stdout.read().split('\n')
        fileRefs = fileRefs[:len(fileRefs) - 1]

        return np.array(fileRefs), None, None

    def get_stats_collector(self, results_output_dir):
        return MultiChannelPlottingDbnTrainingPCAReconstructingStatsCollector(results_output_dir, self.pca_model, 200)
        #return MultiChannelPlottingDbnTrainingStatsCollector(results_output_dir, 400)

