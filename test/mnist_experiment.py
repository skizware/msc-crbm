import cPickle
import gzip
import os
import numpy as np

import crbm

from rbm_trainer import RbmTrainer

learning_rates = [0.007]
target_sparsities = [0.08]
sparsity_constants = [0.9]

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
    for lr in learning_rates:
        for ts in target_sparsities:
            for sc in sparsity_constants:
                recreation_err_squared = []
                np.random.shuffle(train_set)
                print("Running Experiment For Vals {} {} {} - START".format(lr, ts, sc))
                if not os.path.isdir(get_subdir_name(lr, sc, ts)):
                    os.makedirs(get_subdir_name(lr, sc, ts) + '/recreations')
                    os.makedirs(get_subdir_name(lr, sc, ts) + '/histograms')

                    myRbm = crbm.BinaryCrbm(number_of_bases, visible_layer_shape, hidden_layer_shape, sc, ts, lr)
                    trainer = RbmTrainer(myRbm, train_set, valid_set, output_directory='mnist_single_layer/10x10/')

                    iteration_count = 0
                    for image in train_set:
                        hidden_bias_delta, sparsity_delta, \
                        visible_bias_delta, weight_group_delta = trainer.train_given_sample(image)

                        if iteration_count % 500 == 0:
                            test_idx = np.random.randint(1, len(valid_set) - 1)
                            test_sample = valid_set[test_idx].copy()
                            recreation = myRbm.gibbs_vhv(test_sample)
                            trainer.collect_statistics(hidden_bias_delta, iteration_count, recreation_err_squared,
                                                       sparsity_delta, weight_group_delta, 1, test_sample, recreation)

                        iteration_count += 1

                    np.save('{}/learned_state.npy'.format(get_subdir_name(lr, sc, ts)), myRbm.getStateObject())
                print("Running Experiment For Vals {} {} {} - END".format(lr, ts, sc))


def get_subdir_name(lr, sc, ts):
    return 'mnist_single_layer/10x10/' + str(lr) + '_' + str(ts) + '_' + str(sc)


run_experiment()
