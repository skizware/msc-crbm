import numpy as np
from sklearn.externals import joblib
from sklearn.linear_model import SGDClassifier
import dbn
import random
from subprocess import Popen, PIPE

DBN_TO_TEST = '/home/dave/code/msc-crbm/test/v2/timit_batch_32_non_pooled_withPCA_cd1/moresparse_cont_2/layer_0/lr_0.05_st_0.01_slr_0.9/dbn_state_2460.npy'
TRAIN_SET_LOCATION = '/home/dave/code/msc-crbm/test/v2/TIMIT_TRAIN/pre-processed'
TEST_SET_LOCATION = '/home/dave/code/msc-crbm/test/v2/TIMIT_TEST/pre-processed'
PCA_MODEL_LOCATION = '/home/dave/code/msc-crbm/test/v2/TIMIT/pre-processed/principle_components_model.pkl'
MALE_VAL = 0
FEMALE_VAL = 1

pca_freq = joblib.load(PCA_MODEL_LOCATION)
myDbn = dbn.DbnFromStateBuilder.init_dbn(np.load(DBN_TO_TEST).item())
proc = Popen(['find', ('%s' % TRAIN_SET_LOCATION), '-iname', '*.npy'], stdout=PIPE)
all_train_input_refs = proc.stdout.read().split('\n')
all_train_input_refs = all_train_input_refs[:len(all_train_input_refs) - 1]

proc = Popen(['find', ('%s' % TEST_SET_LOCATION), '-iname', '*.npy'], stdout=PIPE)
all_test_input_refs = proc.stdout.read().split('\n')
all_test_input_refs = all_test_input_refs[:len(all_test_input_refs) - 1]

linear_model = SGDClassifier()


def train_sgd(model_to_train, train_npy_refs, the_dbn, pca_model):
    file_count = 0
    data = []
    label = []
    for file_loc in train_npy_refs:
        file_count += 1
        specgram = np.load(file_loc)[0]
        specgram_reduced = pca_model.transform(specgram.swapaxes(0, 1))
        features = the_dbn.infer_hid_given_vis([[specgram_reduced]])[0]
        features = features.reshape(features.size)
        data.append(features)
        theLabel = MALE_VAL if file_loc[file_loc.rfind('/DR') + 5:file_loc.rfind('/DR') + 6] is 'M' else FEMALE_VAL
        label.append(theLabel)

        if file_count % 500 is 0:
            print "Partially fitting 500 recs"
            data = np.asarray(data)
            label = np.asarray(label)
            model_to_train.partial_fit(data, label, classes=[0, 1])
            data = []
            label = []


def verify_sgd(model_to_verify, test_npy_refs, the_dbn, pca_model):
    data = []
    label = []
    for file_loc in test_npy_refs:
        print file_loc
        specgram = np.load(file_loc)[0]
        specgram_reduced = pca_model.transform(specgram.swapaxes(0, 1))
        features = the_dbn.infer_hid_given_vis([[specgram_reduced]])[0]
        features = features.reshape(features.size)
        data.append(features)
        theLabel = MALE_VAL if file_loc[file_loc.rfind('/DR') + 5:file_loc.rfind('/DR') + 6] is 'M' else FEMALE_VAL
        label.append(theLabel)

    data = np.asarray(data)
    label = np.asarray(label)

    predictions = model_to_verify.predict(data)

    result = np.abs(predictions - label)
    num_wrong = np.count_nonzero(result)
    class_error = num_wrong / float(len(result))

    print "Class error is = {}".format(class_error)


for epoch_num in xrange(0, 5):
    train_sgd(linear_model, all_train_input_refs, myDbn, pca_freq)
    verify_sgd(linear_model, all_test_input_refs, myDbn, pca_freq)
    random.shuffle(all_train_input_refs)
