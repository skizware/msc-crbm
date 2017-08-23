import dbn
import numpy as np
from sklearn.externals import joblib
from experiment import TimitExperiment
from experiment import KEY_VIS_SHAPE, KEY_POOL_RATIO, KEY_HID_SHAPE, KEY_RE_TRAIN_TOP_LAYER, KEY_TARGET_SPARSITIES, KEY_LEARNING_RATES, KEY_SPARSITY_LEARNING_RATES, KEY_BATCH_SIZE, KEY_NUM_EPOCHS

DIR_OUT_RESULTS = 'results/timit/300feat-scaled-pca-norandreconstr/pooled/'
starting_dbn = dbn.GaussianVisibleDbn()
# starting_dbn = dbn.DbnFromStateBuilder.init_dbn(np.load('/home/dave/code/msc-crbm/test/v2/tmp/1d/300feat-scaled-pcad/layer_0/lr_0.001_st_1.0_slr_0.0/dbn_state.npy').item())

pca_model = joblib.load('/home/dave/code/msc-crbm/test/v2/PCA_MODEL/scaled/pca_model.pkl')


def inverse_transform(components, transformed_data):
    return np.dot(transformed_data, components)


#pca_model.inverse_transform = inverse_transform
myScaler = joblib.load('/home/dave/code/msc-crbm/test/v2/scaler/scaler_dump.pkl')
timitExp = TimitExperiment(starting_dbn, DIR_OUT_RESULTS, pca_model=pca_model, scale_features=myScaler)

grids_example = {
    KEY_VIS_SHAPE: (1, 80, 1, 92),
    KEY_HID_SHAPE: (1, 300, 1, 87),
    KEY_POOL_RATIO: (1, 3),
    KEY_LEARNING_RATES: [0.008],
    KEY_TARGET_SPARSITIES: [0.03],
    KEY_SPARSITY_LEARNING_RATES: [0.8],
    KEY_BATCH_SIZE: 16,
    KEY_NUM_EPOCHS: 200,
    KEY_RE_TRAIN_TOP_LAYER: False
}

resultant2 = timitExp.run_grids(grids_example)
