import dbn
import numpy as np
from sklearn.externals import joblib
from experiment import TimitExperiment
from experiment import KEY_VIS_SHAPE, KEY_POOL_RATIO, KEY_HID_SHAPE, KEY_RE_TRAIN_TOP_LAYER, KEY_TARGET_SPARSITIES, KEY_LEARNING_RATES, KEY_SPARSITY_LEARNING_RATES, KEY_BATCH_SIZE, KEY_NUM_EPOCHS

DIR_OUT_RESULTS = 'tmp/1d/non-scaled-but-pcad/'
starting_dbn = dbn.GaussianVisibleDbn()
# starting_dbn = dbn.DbnFromStateBuilder.init_dbn(np.load('/home/dave/code/msc-crbm/test/v2/tmp/1d/layer_0/lr_0.003_st_0.03_slr_0.9/dbn_state_22550.npy').item())

pca_model = joblib.load('/home/dave/code/msc-crbm/test/v2/TIMIT/pre-processed/scaled/principle_components_model.pkl')
timitExp = TimitExperiment(starting_dbn, DIR_OUT_RESULTS, pca_model, scale_features=True)

grids_example = {
    KEY_VIS_SHAPE: (1, 80, 1, 92),
    KEY_HID_SHAPE: (1, 200, 1, 87),
    KEY_POOL_RATIO: 0,
    KEY_LEARNING_RATES: [0.003],
    KEY_TARGET_SPARSITIES: [0.03],
    KEY_SPARSITY_LEARNING_RATES: [0.9],
    KEY_BATCH_SIZE: 32,
    KEY_NUM_EPOCHS: 96,
    KEY_RE_TRAIN_TOP_LAYER: False
}

resultant2 = timitExp.run_grids(grids_example)
