import numpy as np
from sklearn.externals import joblib
import dbn
import random
from verify import TimitSGDClassificationDbnVerifier
from data import NumpyArrayStftMagnitudeDataLoader, TimitGenderLabelResolver, TimitSpeakerLabelResolver
from subprocess import Popen, PIPE

DBN_TO_TEST = '/home/dave/code/msc-crbm/test/v2/results/timit/300feat-scaled-pca-norandreconstr/layer_0/lr_0.008_st_0.03_slr_0.8/dbn_state_41871.npy'
DBN_TO_TEST_LOCATION = '/home/dave/code/msc-crbm/test/v2/results/timit/hand-picked/pooled/300feat-scaled-pca-norandreconstr/layer_0/lr_0.001_st_0.05_slr_0.1/'
TRAIN_SET_LOCATION = '/home/dave/code/msc-crbm/test/v2/TIMIT_TRAIN_NO_CHUNK_SPEAKER_ID/pre-processed/'
TEST_SET_LOCATION = '/home/dave/code/msc-crbm/test/v2/TIMIT_TEST_NO_CHUNK_SPEAKER_ID/pre-processed/'
PCA_MODEL_LOCATION = '/home/dave/code/msc-crbm/test/v2/PCA_MODEL/scaled/pca_model.pkl'
SCALER_LOCATION = '/home/dave/code/msc-crbm/test/v2/scaler/scaler_dump.pkl'

pca_freq = joblib.load(PCA_MODEL_LOCATION)
myScaler = joblib.load(SCALER_LOCATION)

proc = Popen(['find', ('%s' % TRAIN_SET_LOCATION), '-iname', '*.npy'], stdout=PIPE)
all_train_input_refs = proc.stdout.read().split('\n')
all_train_input_refs = all_train_input_refs[:len(all_train_input_refs) - 1]
random.shuffle(all_train_input_refs)

proc = Popen(['find', ('%s' % TEST_SET_LOCATION), '-iname', '*.npy'], stdout=PIPE)
all_test_input_refs = proc.stdout.read().split('\n')
all_test_input_refs = all_test_input_refs[:len(all_test_input_refs) - 1]
random.shuffle(all_test_input_refs)

label_resolver = TimitSpeakerLabelResolver()
data_loader = NumpyArrayStftMagnitudeDataLoader(pca_freq, label_resolver, scale_features=myScaler)

class_labels = set()
for train_label_ref in all_train_input_refs:
    class_labels.add(label_resolver.load_label(train_label_ref))

class_labels = list(class_labels)

proc = Popen(['find', DBN_TO_TEST_LOCATION, '-iname', 'dbn_state.npy'], stdout=PIPE)
dbn_refs = proc.stdout.read().split('\n')
dbn_refs = dbn_refs[:len(dbn_refs) - 1]

for dbn_ref in dbn_refs:
    out_dir  = dbn_ref[:dbn_ref.index('dbn_state.npy')]
    myDbn = dbn.DbnFromStateBuilder.init_dbn(np.load(dbn_ref).item())
    verification_exp = TimitSGDClassificationDbnVerifier(myDbn, [all_train_input_refs, all_train_input_refs],
                                                         [all_test_input_refs, all_test_input_refs], class_labels, data_loader,
                                                         time_bins_per_subsample=8, train_batch_size=None,
                                                         num_train_epochs=100)

    classification_errors_per_epoch = verification_exp.verify_model()

    print classification_errors_per_epoch
    with open(out_dir + 'classification_result.txt', 'w+') as f:
        for classification_result in classification_errors_per_epoch:
            f.write("Classification error =  {}".format(classification_result))
            f.write("\n")

print "FIN!"
