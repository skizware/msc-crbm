import cPickle
import gzip
import os
import numpy as np
from sklearn import svm
import crbm
import cdbn

from dbn_trainer import DbnTrainer

learning_rates = [0.01]
target_sparsities = [0.1]
sparsity_constants = [0.1,0.9]

number_of_bases = 100
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
                np.random.shuffle(train_set)
                print("Running Experiment For Vals {} {} {} - START".format(lr, ts, sc))
                if not os.path.isdir(get_subdir_name(lr, sc, ts)):
                    layer1 = crbm.BinaryCrbm(number_of_bases, visible_layer_shape, hidden_layer_shape, sc, ts, lr)
                    stateObj = np.load('/home/dave/dev/MSc/crbm/test/mnist_single_layer/cross_validated/10x10/0.01_0.1_0.1/learned_state.npy')
                    layer1.loadStateObject(stateObj.item())

                    layer2 = crbm.BinaryCrbm(100, (1, 40, 19, 19), (1, 1, 10, 10), sc, ts, lr)
                    layers = [layer1, layer2]
                    myDbn = cdbn.Dbn(layers)
                    trainer = DbnTrainer(myDbn, train_set, valid_set, output_directory='mnist_dbn_2layer/cross_validated/10x10x40_10x10x100/')

                    #os.makedirs(get_subdir_name(lr, sc, ts, 0) + '/recreations')
                    #os.makedirs(get_subdir_name(lr, sc, ts, 0) + '/histograms')
                    os.makedirs(get_subdir_name(lr, sc, ts, 1) + '/recreations')
                    os.makedirs(get_subdir_name(lr, sc, ts, 1) + '/histograms')

                    trainer.train_dbn_unsupervised(starting_layer=1)
                    np.save(get_subdir_name(lr, sc, ts, 1) + '/learned_state.npy', myDbn.getStateObject())

                    svc_train_set = []
                    svc_test_set = []
                    svc_train_labels = test_set[1][:test_set[1].shape[0] / 2]
                    svc_test_labels = test_set[1][test_set[1].shape[0] / 2:]
                    for svc_training_image in test_set[0][:test_set[0].shape[0] / 2]:
                        im = svc_training_image.reshape((1, 1, 28, 28))
                        features = myDbn.sample_h_given_v(im)
                        svc_train_set.append(features[1].reshape(features[1].size))

                    for svc_testing_image in test_set[0][test_set[0].shape[0] / 2:]:
                        im = svc_testing_image.reshape((1, 1, 28, 28))
                        features = myDbn.sample_h_given_v(im)
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


def get_subdir_name(lr, sc, ts, layerNum=1):
    return 'mnist_dbn_2layer/cross_validated/10x10x40_10x10x100/layer_{}/'.format(layerNum) + str(lr) + '_' + str(ts) + '_' + str(sc)


run_experiment()
