import cPickle
import gzip
import os
import numpy as np

import crbm
import cdbn

from dbn_trainer import DbnTrainer

learning_rates = [0.007]
target_sparsities = [0.05]
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
                np.random.shuffle(train_set)
                print("Running Experiment For Vals {} {} {} - START".format(lr, ts, sc))
                if not os.path.isdir(get_subdir_name(lr, sc, ts)):
                    layer1 = crbm.BinaryCrbm(number_of_bases, visible_layer_shape, hidden_layer_shape, sc, ts, lr)
                    stateObj = np.load('/home/dave/dev/MSc/crbm/test/mnist_single_layer/10x10/0.007_0.08_0.9/learned_state.npy')
                    layer1.loadStateObject(stateObj.item())

                    layer2 = crbm.BinaryCrbm(100, (1, 40, 19, 19), (1, 1, 10, 10), sc, 0.03, lr)
                    layers = [layer1, layer2]
                    myDbn = cdbn.Dbn(layers)
                    trainer = DbnTrainer(myDbn, train_set, valid_set, output_directory='mnist_dbn_2layer/10x10x40_10x10x100/')

                    #os.makedirs(get_subdir_name(lr, sc, ts, 0) + '/recreations')
                    #os.makedirs(get_subdir_name(lr, sc, ts, 0) + '/histograms')
                    os.makedirs(get_subdir_name(lr, sc, 0.03, 1) + '/recreations')
                    os.makedirs(get_subdir_name(lr, sc, 0.03, 1) + '/histograms')

                    trainer.train_dbn_unsupervised(starting_layer=1)
                    np.save(get_subdir_name(lr, sc, 0.03, 1) + '/../learned_state.npy', myDbn.getStateObject())
                print("Running Experiment For Vals {} {} {} - END".format(lr, ts, sc))


def get_subdir_name(lr, sc, ts, layerNum=1):
    return 'mnist_dbn_2layer/10x10x40_10x10x100/layer_{}/'.format(layerNum) + str(lr) + '_' + str(ts) + '_' + str(sc)


run_experiment()
