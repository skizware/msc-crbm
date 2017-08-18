import numpy as np
from sklearn.externals import joblib
import dbn
import random
from verify import TimitSGDClassificationDbnVerifier
from data import NumpyArrayStftMagnitudeDataLoader, TimitGenderLabelResolver, TimitSpeakerLabelResolver
from subprocess import Popen, PIPE

DBN_TO_TEST = '/home/dave/code/msc-crbm/test/v2/results/timit/300feat-noscale-norandreconstr/layer_0/lr_0.008_st_0.03_slr_0.8/dbn_state_136286.npy'
TRAIN_SET_LOCATION = '/home/dave/code/msc-crbm/test/v2/TIMIT_TRAIN_NO_CHUNK_SPEAKER_ID/pre-processed/'
TEST_SET_LOCATION = '/home/dave/code/msc-crbm/test/v2/TIMIT_TEST_NO_CHUNK_SPEAKER_ID/pre-processed/'
PCA_MODEL_LOCATION = '/home/dave/code/msc-crbm/test/v2/PCA_MODEL/pca_model.pkl'
SCALER_LOCATION = '/home/dave/code/msc-crbm/test/v2/scaler/scaler_dump.pkl'

pca_freq = joblib.load(PCA_MODEL_LOCATION)
myScaler = joblib.load(SCALER_LOCATION)
myDbn = dbn.DbnFromStateBuilder.init_dbn(np.load(DBN_TO_TEST).item())

proc = Popen(['find', ('%s' % TRAIN_SET_LOCATION), '-iname', '*.npy'], stdout=PIPE)
all_train_input_refs = proc.stdout.read().split('\n')
all_train_input_refs = all_train_input_refs[:len(all_train_input_refs) - 1]
random.shuffle(all_train_input_refs)

proc = Popen(['find', ('%s' % TEST_SET_LOCATION), '-iname', '*.npy'], stdout=PIPE)
all_test_input_refs = proc.stdout.read().split('\n')
all_test_input_refs = all_test_input_refs[:len(all_test_input_refs) - 1]
random.shuffle(all_test_input_refs)

label_resolver = TimitSpeakerLabelResolver()
data_loader = NumpyArrayStftMagnitudeDataLoader(pca_freq, label_resolver, scale_features=None)

class_labels = set()
for train_label_ref in all_train_input_refs:
    class_labels.add(label_resolver.load_label(train_label_ref))

class_labels = list(class_labels)


verification_exp = TimitSGDClassificationDbnVerifier(myDbn, [all_train_input_refs, all_train_input_refs],
                                                     [all_test_input_refs, all_test_input_refs], class_labels, data_loader,
                                                     time_bins_per_subsample=25, train_batch_size=None,
                                                     num_train_epochs=400)

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
