import cPickle, gzip
import pylab as plt
import numpy as np
import crbm
from layer import BinaryVisibleNonPooledLayer
from datetime import datetime

BATCH_SIZE = 1

myLayer = None

def load_data():
    f = gzip.open("/home/dave/Downloads/mnist.pkl.gz", "rb")
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    return [train_set, valid_set, test_set]


def testRbm(theLayer):
    train_set, _, _ = load_data()
    train_set = train_set[0].reshape(50000, 1, 28, 28)

    for index in range(0, train_set.shape[0]/ BATCH_SIZE):
        if(index % 1000 == 0):
            print("Training on sample {} at time {}".format(index * BATCH_SIZE, datetime.now()))
        batch = train_set[index * BATCH_SIZE:(index + 1) * BATCH_SIZE, :, :, :]
        theLayer.train_on_minibatch(batch)

    print "Training ended at {}".format(datetime.now())

myLayer = BinaryVisibleNonPooledLayer((1, 1, 28, 28), (1, 40, 19, 19), target_sparsity=0.1,
                                          sparsity_learning_rate=0.1)
testRbm(myLayer)
