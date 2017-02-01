import sys
import cPickle, gzip
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, '../src')
import crbm
from ast import literal_eval as make_tuple


def testRbm(myCrbm):
    f = gzip.open("/home/dave/Downloads/mnist.pkl.gz", "rb")
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()

    weight_updates = []
    hidden_bias_updates = []
    sparsity_updates = []
    visible_bias_updates = []
    recreation_err_sqrd = []

    for index in range(0, train_set[0].shape[0]):

        print "Test #" + str(index)
        imgMat = np.reshape(train_set[0][index].copy(), (28, 28))
        imgMat = imgMat - np.mean(imgMat)
        imgMat = imgMat / np.std(imgMat)
        weight_group_delta, hidden_bias_delta, \
        sparsity_delta, bias_updates, visible_bias_delta = myCrbm.contrastive_divergence(np.array([[imgMat]]))

        if index % 500 == 0:
            collect_statistics(hidden_bias_delta, hidden_bias_updates, index, recreation_err_sqrd, sparsity_delta,
                               sparsity_updates, valid_set, visible_bias_delta, visible_bias_updates,
                               weight_group_delta, weight_updates)

    return [weight_updates, hidden_bias_updates, sparsity_updates, visible_bias_updates, recreation_err_sqrd]


def collect_statistics(hidden_bias_delta, hidden_bias_updates, index, recreation_err_sqrd, sparsity_delta,
                       sparsity_updates, valid_set, visible_bias_delta, visible_bias_updates, weight_group_delta,
                       weight_updates):
    add_param_update_metrics_to_collectors(hidden_bias_delta, hidden_bias_updates, sparsity_delta, sparsity_updates,
                                           visible_bias_delta, visible_bias_updates, weight_group_delta, weight_updates)

    test_idx = np.random.randint(1, len(valid_set[0]) - 1)
    testSample = valid_set[0][test_idx].copy()

    recreation, testSample = get_recreation(testSample)
    recreation_err_sqrd.append(((testSample - recreation[0]) ** 2).sum())
    plot_and_save_sample_recreation_comparison(index, recreation, testSample)

    fig = plt.figure()
    fig.add_subplot(1, 1, 1).set_title('Recreation squared error')
    plt.plot(recreation_err_sqrd)
    fig.savefig("recreation_squared_error.png")


def plot_and_save_sample_recreation_comparison(index, recreation, testSample):
    fig = plt.figure()
    orig = fig.add_subplot(1, 2, 1)
    orig.set_title('Original')
    plt.imshow(testSample[0][0], cmap='gray')
    recreat = fig.add_subplot(1, 2, 2)
    recreat.set_title('Recreation')
    plt.imshow(recreation[0][0][0], cmap='gray')
    fig.savefig('recreations/recreation_after_' + str(index) + '_iterations.png')


def add_param_update_metrics_to_collectors(hidden_bias_delta, hidden_bias_updates, sparsity_delta, sparsity_updates,
                                           visible_bias_delta, visible_bias_updates, weight_group_delta,
                                           weight_updates):
    weight_updates.append(np.average(weight_group_delta, (1, 2, 3)))
    hidden_bias_updates.append(hidden_bias_delta[0])
    sparsity_updates.append(sparsity_delta)
    visible_bias_updates.append(visible_bias_delta)


def get_recreation(testSample):
    testSample = testSample.reshape((28, 28))
    testSample = testSample - testSample.mean()
    testSample = testSample / testSample.std()
    testSample = np.array([[testSample]])
    recreation = myRbm.gibbs_vhv(testSample)
    return recreation, testSample


if __name__ == "__main__":
    if len(sys.argv) < 7:
        print("Incorrect number of args - need 6")
        sys.exit(2)

    print sys.argv

    visible_layer_shape = make_tuple(sys.argv[1])
    hidden_layer_shape = make_tuple(sys.argv[2])
    learning_rate = float(sys.argv[3])
    target_sparsity = float(sys.argv[4])
    sparsity_regularization_constant = float(sys.argv[5])
    num_bases = int(sys.argv[6])
    myRbm = crbm.crbm(num_bases, visible_layer_shape, hidden_layer_shape, sparsity_regularization_constant,
                      target_sparsity,
                      learning_rate)
    testRbm(myRbm)
