import dbn
import numpy as np
from sklearn.externals import joblib
from experiment import TimitExperiment
from experiment import KEY_VIS_SHAPE, KEY_POOL_RATIO, KEY_HID_SHAPE, KEY_RE_TRAIN_TOP_LAYER, KEY_TARGET_SPARSITIES, KEY_LEARNING_RATES, KEY_SPARSITY_LEARNING_RATES, KEY_BATCH_SIZE, KEY_NUM_EPOCHS

DIR_OUT_RESULTS = 'timit_batch_32_non_pooled_withPCA_cd1/moresparse_cont_2/'
#starting_dbn = dbn.GaussianVisibleDbn()
starting_dbn = dbn.DbnFromStateBuilder.init_dbn(np.load('/home/dave/code/msc-crbm/test/v2/timit_batch_32_non_pooled_withPCA_cd1/moresparse_cont/layer_0/lr_0.05_st_0.01_slr_0.9/dbn_state_3690.npy').item())

pca_model = joblib.load('/home/dave/code/msc-crbm/test/v2/TIMIT/pre-processed/principle_components_model.pkl')
timitExp = TimitExperiment(starting_dbn, DIR_OUT_RESULTS, pca_model)

grids_example = {
    KEY_VIS_SHAPE: (1, 1, 92, 80),
    KEY_HID_SHAPE: (1, 200, 87, 75),
    KEY_POOL_RATIO: 0,
    KEY_LEARNING_RATES: [0.05],
    KEY_TARGET_SPARSITIES: [0.01],
    KEY_SPARSITY_LEARNING_RATES: [0.9],
    KEY_BATCH_SIZE: 32,
    KEY_NUM_EPOCHS: 96,
    KEY_RE_TRAIN_TOP_LAYER: True
}

resultant2 = timitExp.run_grids(grids_example)
