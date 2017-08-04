import numpy as np
from sklearn.externals import joblib
from sklearn.linear_model import SGDClassifier
import dbn
import random
from verify import TimitSGDClassificationDbnVerifier
from data import NumpyArrayStftMagnitudeDataLoader, TimitGenderLabelResolver
from subprocess import Popen, PIPE

DBN_TO_TEST = '/home/dave/code/msc-crbm/test/v2/tmp/1d/layer_0/lr_0.001_st_0.03_slr_0.9/dbn_state_20500.npy'
TRAIN_SET_LOCATION = '/home/dave/code/msc-crbm/test/v2/TIMIT_TRAIN_NO_CHUNK/pre-processed'
TEST_SET_LOCATION = '/home/dave/code/msc-crbm/test/v2/TIMIT_TEST_NO_CHUNK/pre-processed'
PCA_MODEL_LOCATION = '/home/dave/code/msc-crbm/test/v2/TIMIT/pre-processed/scaled/principle_components_model.pkl'
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

data_loader = NumpyArrayStftMagnitudeDataLoader(pca_freq, TimitGenderLabelResolver(), scale_features=True)

verification_exp = TimitSGDClassificationDbnVerifier(myDbn, [all_train_input_refs, all_train_input_refs],
                                                     [all_test_input_refs, all_test_input_refs], [0, 1], data_loader,
                                                     time_bins_per_subsample=100, train_batch_size=500,
                                                     num_train_epochs=5)

all_predicted_data_points = verification_exp.verify_model()

for epoch_data_points in all_predicted_data_points:
    classified_right = 0
    classified_wrong = 0
    for data_point in epoch_data_points:
        if data_point.class_label == data_point.predicted_label:
            classified_right += 1
        else:
            classified_wrong += 1

    print "Classification Error % = {}".format((classified_wrong / float(len(epoch_data_points))) * 100)

print "FIN!"
