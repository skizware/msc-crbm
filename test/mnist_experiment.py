import cPickle
import gzip
import os
import numpy as np

import crbm

from rbm_trainer import RbmTrainer

starting_learning_rate = 0.008
starting_target_sparsity = 0.
starting_sparsity_constant = 0.

lr_magnitude_multiplier = 0.1
ts_magnitude_multiplier = 0.9
sc_magnitude_multiplier = 0.1

lr_num_values_to_try = 1
ts_num_values_to_try = 1
sc_num_values_to_try = 1

number_of_bases = 40
visible_layer_shape = (1, 1, 28, 28)
hidden_layer_shape = (1, 1, 19, 19)

print("Loading Training Set - START")
f = gzip.open("/home/dave/Downloads/mnist.pkl.gz", "rb")
train_set, valid_set, test_set = cPickle.load(f)
f.close()
print("Loading Training Set - END")

print("Initializing Training Set - START")
train_set = train_set[0].copy()
train_set = train_set.reshape((50000, 1, 1, 28, 28))
valid_set = valid_set[0].copy()
valid_set = valid_set.reshape((10000, 1, 1, 28, 28))
print("Initializing Training Set - END")


def run_experiment():
    for i in range(0, lr_num_values_to_try):
        for j in range(0, ts_num_values_to_try):
            for k in range(0, sc_num_values_to_try):
                np.random.shuffle(train_set)
                lr = starting_learning_rate * lr_magnitude_multiplier ** i
                ts = starting_target_sparsity * ts_magnitude_multiplier ** j
                sc = starting_sparsity_constant * sc_magnitude_multiplier ** k
                print("Running Experiment For Vals {} {} {} - START".format(lr, ts, sc))
                if not os.path.isdir(get_subdir_name(lr, sc, ts)):
                    os.makedirs(get_subdir_name(lr, sc, ts) + '/recreations')
                    os.makedirs(get_subdir_name(lr, sc, ts) + '/histograms')

                    myRbm = crbm.BinaryCrbm(number_of_bases, visible_layer_shape, hidden_layer_shape, sc, ts, lr)
                    trainer = RbmTrainer(myRbm, train_set, valid_set)
                    trainer.train_rbm_unsupervised()
                    print("Running Experiment For Vals {} {} {} - END".format(lr, ts, sc))


def get_subdir_name(lr, sc, ts):
    return str(lr) + '_' + str(ts) + '_' + str(sc)


run_experiment()
