import cPickle, gzip
import pylab as plt
import numpy as np
import crbm
from layer import BinaryVisibleNonPooledLayer
from datetime import datetime


def load_data():
    f = gzip.open("/home/dave/Downloads/mnist.pkl.gz", "rb")
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    return [train_set, valid_set, test_set]


def testRbm():
    train_set, _, _ = load_data()
    myLayer = BinaryVisibleNonPooledLayer((1, 1, 28, 28), (1, 40, 19, 19), target_sparsity=0.1,
                                          sparsity_learning_rate=0.1)
    for index in range(0, train_set[0].shape[0]):
        if (index % 1000 == 0):
            print("Training on sample {} at time {}".format(index, datetime.now()))
        imgMat = np.reshape(train_set[0][index], (1, 1, 28, 28))
        myLayer.train_on_minibatch(imgMat)

    print "Training ended at {}".format(datetime.now())


testRbm()