import dbn
import numpy as np
from sklearn.externals import joblib
from experiment import TimitExperiment
from experiment import KEY_VIS_SHAPE, KEY_POOL_RATIO, KEY_HID_SHAPE, KEY_RE_TRAIN_TOP_LAYER, KEY_TARGET_SPARSITIES, KEY_LEARNING_RATES, KEY_SPARSITY_LEARNING_RATES, KEY_BATCH_SIZE, KEY_NUM_EPOCHS

DIR_OUT_RESULTS = 'results/timit/hand-picked/pooled/chain-sampled/300feat-scaled-pca/'
starting_dbn = dbn.GaussianVisibleDbnPersistentChainSampling()
# starting_dbn = dbn.DbnFromStateBuilder.init_dbn(np.load('/home/dave/code/msc-crbm/test/v2/results/timit/annealed/300feat-scaled-pca-norandreconstr/layer_0/lr_0.004_st_0.025_slr_0.7/dbn_state.npy').item())

pca_model = joblib.load('/home/dave/code/msc-crbm/test/v2/PCA_MODEL/scaled/pca_model.pkl')
myScaler = joblib.load('/home/dave/code/msc-crbm/test/v2/scaler/scaler_dump.pkl')
timitExp = TimitExperiment(starting_dbn, DIR_OUT_RESULTS, pca_model=pca_model, scale_features=myScaler)

grids_example = {
    KEY_VIS_SHAPE: (1, 80, 1, 92),
    KEY_HID_SHAPE: (1, 300, 1, 87),
    KEY_POOL_RATIO: (1, 3),
    KEY_LEARNING_RATES: [0.001],
    KEY_TARGET_SPARSITIES: [0.1],
    KEY_SPARSITY_LEARNING_RATES: [0.1],
    KEY_BATCH_SIZE: 16,
    KEY_NUM_EPOCHS: 100,
    KEY_RE_TRAIN_TOP_LAYER: False
}

resultant2 = timitExp.run_grids(grids_example)
