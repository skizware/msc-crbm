import cPickle, gzip
import pylab as plt
import numpy as np
import crbm

f = gzip.open("/home/dave/Downloads/mnist.pkl.gz", "rb")
train_set, valid_set, test_set = cPickle.load(f)
f.close()


def testRbm(myCrbm):
    for index in range(0,train_set[0].shape[0]):
	    print "Test #" + str(index)
	    imgMat = np.reshape(train_set[0][index], (28,28))
	    imgMat = imgMat - np.mean(imgMat)
	    imgMat = imgMat/np.std(imgMat)
	    myCrbm.contrastive_divergence(imgMat)


