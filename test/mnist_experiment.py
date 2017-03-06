import cPickle
import gzip
import os
import numpy as np
from subprocess import call
from sklearn import svm

import crbm

from rbm_trainer import RbmTrainer

learning_rates = [0.01]
target_sparsities = [0.1]
sparsity_constants = [0.1, 0.3, 0.5]

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
                    trainer = RbmTrainer(myRbm, train_set, valid_set, output_directory='mnist_single_layer/cross_validated/10x10/')

                    iteration_count = 0
                    for image in train_set:
                        hidden_bias_delta, sparsity_delta, \
                        visible_bias_delta, weight_group_delta = trainer.train_given_sample(image)

                        if iteration_count % 5000 == 0:
                            test_idx = np.random.randint(1, len(valid_set) - 1)
                            test_sample = valid_set[test_idx].copy()
                            recreation = myRbm.gibbs_vhv(test_sample)
                            trainer.collect_statistics(hidden_bias_delta, iteration_count, recreation_err_squared,
                                                       sparsity_delta, weight_group_delta, 1, test_sample, recreation,
                                                       __unblockshaped(myRbm.get_weight_groups()[:,0,:,:], 50, 80))

                        iteration_count += 1

                    np.save('{}/learned_state.npy'.format(get_subdir_name(lr, sc, ts)), myRbm.getStateObject())

                    svc_train_set = []
                    svc_test_set = []
                    svc_train_labels = test_set[1][:test_set[1].shape[0] / 2]
                    svc_test_labels = test_set[1][test_set[1].shape[0] / 2:]
                    for svc_training_image in test_set[0][:test_set[0].shape[0] / 2]:
                        im = svc_training_image.reshape((1, 1, 28, 28))
                        features = myRbm.sample_h_given_v(im)
                        svc_train_set.append(features[1].reshape(features[1].size))

                    for svc_testing_image in test_set[0][test_set[0].shape[0] / 2:]:
                        im = svc_testing_image.reshape((1, 1, 28, 28))
                        features = myRbm.sample_h_given_v(im)
                        svc_test_set.append(features[1].reshape(features[1].size))

                    svc_train_set = np.array(svc_train_set)
                    svc_test_set = np.array(svc_test_set)
                    svc_train_labels = np.array(svc_train_labels)
                    svc_test_labels = np.array(svc_test_labels)

                    svc = svm.SVC(kernel='linear')
                    svc.fit(svc_train_set, svc_train_labels)
                    predicted = svc.predict(svc_test_set)
                    diff = svc_test_labels - predicted
                    numWrong = np.count_nonzero(diff)
                    classification_error = float(numWrong) / float(svc_test_set.shape[0]) * 100
                    with open('{}/classification_results.txt'.format(get_subdir_name(lr, sc, ts)), 'w') as f:
                        f.write('Classification error = {}'.format(classification_error))

                print("Running Experiment For Vals {} {} {} - END".format(lr, ts, sc))


def get_subdir_name(lr, sc, ts):
    return 'mnist_single_layer/cross_validated/10x10/' + str(lr) + '_' + str(ts) + '_' + str(sc)


def __unblockshaped(arr, h, w):
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

run_experiment()
