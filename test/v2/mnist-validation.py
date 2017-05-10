import dbn
from experiment import MnistExperiment, MnistDataLoader
import numpy as np
from sklearn import linear_model

DBN_TO_TEST = "/home/dave/code/msc-crbm/test/v2/mnist/results/layer_0/lr_0.1_st_0.3_slr_0.1/layer_1/lr_0.001_st_0.1_slr_0.1/dbn_state.npy"
loader = MnistDataLoader()

stateJson = np.load(DBN_TO_TEST).item()
stateJson['type'] = dbn.BinaryVisibleDbn
myDbn = dbn.DbnFromStateBuilder.init_dbn(stateJson)

train_set_refs, valid_set_refs, test_set_refs = MnistExperiment.load_data_sets_and_labels()

classifier = linear_model.SGDClassifier()
train_data_num_samples = train_set_refs[0].shape[0]

test_set = []
for ref in test_set_refs[0]:
    data = loader.load_data(ref)

    layer_0_expectation = myDbn.infer_hid_given_vis(data, end_layer_index_incl=0)[0]
    layer_1_expectation = myDbn.infer_hid_given_vis(data)[0]
    test_data = np.concatenate(
        (layer_0_expectation.reshape(layer_0_expectation.size), layer_1_expectation.reshape(layer_1_expectation.size)))
    test_set.append(test_data)
print "test set loaded"

num_dataset_chunks = 3
num_samples_per_chunk = train_data_num_samples / num_dataset_chunks

for epoch in xrange(0, 3):
    print "epoch {} - START".format(epoch)

    for chunk in xrange(0, num_dataset_chunks):
        train_set = []
        start_index = chunk * num_samples_per_chunk
        end_index = (chunk + 1) * num_samples_per_chunk
        for ref in train_set_refs[0][start_index:end_index]:
            data = loader.load_data(ref)

            layer_0_expectation = myDbn.infer_hid_given_vis(data, end_layer_index_incl=0)[0]
            layer_1_expectation = myDbn.infer_hid_given_vis(data)[0]
            train_data = np.concatenate((layer_0_expectation.reshape(layer_0_expectation.size),
                                         layer_1_expectation.reshape(layer_1_expectation.size)))
            train_set.append(train_data)
        print "train set loaded {}".format(chunk)

        train_set_and_labels = [train_set, train_set_refs[1][start_index:end_index]]
        if chunk is 0:
            classifier.partial_fit(train_set_and_labels[0], train_set_and_labels[1],
                                   classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        else:
            classifier.partial_fit(train_set_and_labels[0], train_set_and_labels[1])

    predictions = classifier.predict(test_set)
    numWrong = np.count_nonzero(predictions - test_set_refs[1])
    classificationError = float(numWrong)/ float(test_set_refs[1].size)
    print "epoch {} - END - classification error = {}".format(epoch, classificationError * 100)
