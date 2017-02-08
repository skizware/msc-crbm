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
            collect_statistics(myCrbm, hidden_bias_delta, hidden_bias_updates, index, recreation_err_sqrd,
                               sparsity_delta,
                               sparsity_updates, valid_set, visible_bias_delta, visible_bias_updates,
                               weight_group_delta, weight_updates)

    return [weight_updates, hidden_bias_updates, sparsity_updates, visible_bias_updates, recreation_err_sqrd]


def collect_statistics(myRbm, hidden_bias_delta, hidden_bias_updates, index, recreation_err_sqrd, sparsity_delta,
                       sparsity_updates, valid_set, visible_bias_delta, visible_bias_updates, weight_group_delta,
                       weight_updates):
    add_param_update_metrics_to_collectors(hidden_bias_delta, hidden_bias_updates, sparsity_delta, sparsity_updates,
                                           visible_bias_delta, visible_bias_updates, weight_group_delta, weight_updates)

    test_idx = np.random.randint(1, len(valid_set[0]) - 1)
    testSample = valid_set[0][test_idx].copy()

    recreation, testSample = get_recreation(myRbm, testSample)
    plot_and_save_sample_recreation_comparison(index, recreation, testSample)
    plot_and_save_recreation_squared_error(recreation, recreation_err_sqrd, testSample)
    plot_and_save_weight_histograms(index, myRbm, weight_group_delta)
    plot_and_save_hidden_bias_histograms(hidden_bias_delta, index, myRbm, sparsity_delta)
    plot_and_save_learned_filters(index, myRbm)


def plot_and_save_learned_filters(index, myRbm):
    fig, _ = plt.subplots()
    fig.tight_layout()
    fig_learned_filters = fig.add_subplot(1, 1, 1)
    fig_learned_filters.set_title("Learned Filters")
    plt.imshow(unblockshaped(myRbm.th_weight_groups.get_value()[:, 0, :, :], 80, 50), cmap='gray')
    fig.savefig(get_subdir_name(learning_rate, sparsity_regularization_constant,
                                target_sparsity) + '/recreations/filters_' + str(index) + '_iterations.png')
    plt.close(fig)


def plot_and_save_hidden_bias_histograms(hidden_bias_delta, index, myRbm, sparsity_delta):
    fig, _ = plt.subplots(3, 1)
    fig.tight_layout()
    fig_hidden_biases = fig.add_subplot(3, 1, 1)
    fig_hidden_biases.set_title("Hidden Biases")
    plt.hist(myRbm.th_hidden_group_biases.get_value(), bins=20)
    fig_hidden_bias_deltas = fig.add_subplot(3, 1, 2)
    fig_hidden_bias_deltas.set_title("Hidden Bias Deltas")
    plt.hist(hidden_bias_delta.reshape(hidden_bias_delta.size), bins=20)
    fig_hidden_bias_deltas = fig.add_subplot(3, 1, 3)
    fig_hidden_bias_deltas.set_title("Sparsity Deltas")
    plt.hist(sparsity_delta.reshape(sparsity_delta.size), bins=20)
    fig.savefig(get_subdir_name(learning_rate, sparsity_regularization_constant,
                                target_sparsity) + '/histograms/histogram_hidden_biases_' + str(
        index) + '_iterations.png')
    plt.close(fig)


def plot_and_save_weight_histograms(index, myRbm, weight_group_delta):
    fig, _ = plt.subplots(2, 1)
    fig.tight_layout()
    fig_weights = fig.add_subplot(2, 1, 1)
    fig_weights.set_title('Weight Values')
    plt.hist(myRbm.th_weight_groups.get_value().reshape(myRbm.th_weight_groups.get_value().size)
             , bins=100)
    fig_deltas = fig.add_subplot(2, 1, 2)
    fig_deltas.set_title('Weight Deltas')
    plt.hist(weight_group_delta.reshape(weight_group_delta.size), bins=100)
    fig.savefig(get_subdir_name(learning_rate, sparsity_regularization_constant,
                                target_sparsity) + '/histograms/histogram_weights_' + str(index) + '_iterations.png')
    plt.close(fig)


def plot_and_save_recreation_squared_error(recreation, recreation_err_sqrd, testSample):
    recreation_err_sqrd.append(((testSample - recreation[0]) ** 2).sum())
    fig = plt.figure()
    fig.add_subplot(1, 1, 1).set_title('Recreation squared error')
    plt.plot(recreation_err_sqrd)
    fig.savefig(get_subdir_name(learning_rate, sparsity_regularization_constant,
                                target_sparsity) + "/recreation_squared_error.png")
    plt.close(fig)


def get_subdir_name(lr, sc, ts):
    return str(lr) + '_' + str(ts) + '_' + str(sc)


def plot_and_save_sample_recreation_comparison(index, recreation, testSample):
    fig = plt.figure()
    orig = fig.add_subplot(1, 2, 1)
    orig.set_title('Original')
    plt.imshow(testSample[0][0], cmap='gray')
    recreat = fig.add_subplot(1, 2, 2)
    recreat.set_title('Recreation')
    plt.imshow(recreation[0][0][0], cmap='gray')
    fig.savefig(get_subdir_name(learning_rate, sparsity_regularization_constant,
                                target_sparsity) + '/recreations/recreation_after_' + str(index) + '_iterations.png')
    plt.close(fig)


def add_param_update_metrics_to_collectors(hidden_bias_delta, hidden_bias_updates, sparsity_delta, sparsity_updates,
                                           visible_bias_delta, visible_bias_updates, weight_group_delta,
                                           weight_updates):
    weight_updates.append(np.average(weight_group_delta, (1, 2, 3)))
    hidden_bias_updates.append(hidden_bias_delta[0])
    sparsity_updates.append(sparsity_delta)
    visible_bias_updates.append(visible_bias_delta)


def get_recreation(myRbm, testSample):
    testSample = testSample.reshape((28, 28))
    testSample = testSample - testSample.mean()
    testSample = testSample / testSample.std()
    testSample = np.array([[testSample]])
    recreation = myRbm.gibbs_vhv(testSample)
    return recreation, testSample


def do_train(in_num_bases, in_visible_layer_shape, in_hidden_layer_shape, in_sparsity_regularization_constant,
             in_target_sparsity, in_learning_rate):
    myRbm = crbm.crbm(in_num_bases, in_visible_layer_shape, in_hidden_layer_shape, in_sparsity_regularization_constant,
                      in_target_sparsity, in_learning_rate)
    testRbm(myRbm)


def unblockshaped(arr, h, w):
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
    do_train(num_bases, visible_layer_shape, hidden_layer_shape, sparsity_regularization_constant,
             target_sparsity, learning_rate)
