import cdbn
import crbm
import os
import math
from dbn_trainer import DbnTrainer
import random
import numpy as np
import sys

from image_loader import NormalizingResizingImageLoader

LEARNED_STATE_KEY = 'learned_state'
NUM_BASES_KEY = 'num_bases'
OUTPUT_SHAPE_KEY = 'output_shape'
INPUT_SHAPE_KEY = 'input_shape'

DATA_LOCATION = 'data/101_ObjectCategories/'
OUTPUT_LOACTION = 'caltech_subset/'
LAYER_AND_GRID_SUBDIR = 'layer_{}/bases{}_filtW{}_filtH{}/lr{}_ts{}_sc{}/'

IMAGE_LOADER = NormalizingResizingImageLoader(300, 200)
GRIDS = [
    {'lr': [0.001], 'ts': [0.01], 'sc': [0.9], 'lr_error_multiplier':1e-7, 'num_epochs':4}
]

with os.popen('find {} -name *.jpg'.format(DATA_LOCATION)) as f:
    image_refs_unsupervised = f.read().split('\n')

image_refs_unsupervised = image_refs_unsupervised[:len(image_refs_unsupervised) - 1]
random.shuffle(image_refs_unsupervised)

LAYERS = [
    {INPUT_SHAPE_KEY: (1, 1, 200, 300), OUTPUT_SHAPE_KEY: (1, 1, 191, 291), NUM_BASES_KEY: 24, LEARNED_STATE_KEY: None}
]


def get_layer_and_grid_out_dir(targetRbm):
    return LAYER_AND_GRID_SUBDIR.format(layer_idx, targetRbm.numBases,
                                        targetRbm.get_weight_groups().shape[2],
                                        targetRbm.get_weight_groups().shape[3],
                                        targetRbm.get_learning_rate(),
                                        targetRbm.get_target_sparsity(),
                                        targetRbm.get_regularization_rate())


def get_saved_rbm_state(output_dir):
    return np.load(output_dir + 'learned_state.npy').item()


for grid in GRIDS:
    for lr in grid['lr']:
        for ts in grid['ts']:
            for sc in grid['sc']:

                output_dir = OUTPUT_LOACTION
                dbn_layers = []
                layer_idx = 0
                try:
                    for layer in LAYERS:
                        train_current_layer = True
                        train_set = np.array(image_refs_unsupervised[:int(math.ceil(len(image_refs_unsupervised) * 0.8))])
                        valid_set = np.array(image_refs_unsupervised[int(math.ceil(len(image_refs_unsupervised) * 0.8)):])
                        print "Train set size = " + str(train_set.shape)

                        layer_crbm = crbm.crbm(visible_layer_shape=layer[INPUT_SHAPE_KEY],
                                               hidden_layer_shape=layer[OUTPUT_SHAPE_KEY],
                                               numBases=layer[NUM_BASES_KEY], sparsity_regularizarion_rate=sc,
                                               target_sparsity=ts,
                                               learning_rate=lr)

                        output_dir += get_layer_and_grid_out_dir(layer_crbm)

                        if not os.path.isdir(output_dir):
                            os.makedirs(output_dir + 'results/')
                        else:
                            layer_crbm.loadStateObject(get_saved_rbm_state(output_dir))
                            train_current_layer = False

                        dbn_layers.append(layer_crbm)

                        if train_current_layer:
                            myDbn = cdbn.Dbn(dbn_layers)
                            trainer = DbnTrainer(myDbn, train_set, valid_set,
                                                 output_directory=output_dir + 'results/', image_loader=IMAGE_LOADER,
                                                 stats_collection_period=500, num_training_epochs=grid['num_epochs'], lr_error_multiplier=grid['lr_error_multiplier'])
                            trainer.train_dbn_unsupervised(starting_layer=layer_idx)

                            np.save(output_dir + 'learned_state.npy',
                                    layer_crbm.getStateObject())

                        layer_idx += 1
                except:
                    with open(output_dir + 'error.txt', 'w') as f:
                        f.write("Error occurred - " + str(sys.exc_info()[0]))
                        np.save(output_dir + 'dbn_at_error.npy', myDbn.getStateObject())