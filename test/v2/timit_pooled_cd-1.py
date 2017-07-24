import dbn
import numpy as np
from experiment import TimitExperiment
from experiment import KEY_VIS_SHAPE, KEY_POOL_RATIO, KEY_HID_SHAPE, KEY_RE_TRAIN_TOP_LAYER, KEY_TARGET_SPARSITIES, KEY_LEARNING_RATES, KEY_SPARSITY_LEARNING_RATES, KEY_BATCH_SIZE, KEY_NUM_EPOCHS

DIR_OUT_RESULTS = 'tmp/results/'
#starting_dbn = dbn.GaussianVisibleDbn()
starting_dbn = dbn.DbnFromStateBuilder.init_dbn(np.load('/home/dave/code/msc-crbm/test/v2/timit_batch_4_pooled_cd1/results/layer_0/lr_0.005_st_0.03_slr_0.9/dbn_state_6574.npy').item())
timitExp = TimitExperiment(starting_dbn, DIR_OUT_RESULTS)

grids_example = {
    KEY_VIS_SHAPE: (1, 1, 257, 115),
    KEY_HID_SHAPE: (1, 128, 252, 110),
    KEY_POOL_RATIO: 2,
    KEY_LEARNING_RATES: [0.005],
    KEY_TARGET_SPARSITIES: [0.03],
    KEY_SPARSITY_LEARNING_RATES: [0.9],
    KEY_BATCH_SIZE: 4,
    KEY_NUM_EPOCHS: 96,
    KEY_RE_TRAIN_TOP_LAYER: True
}

resultant2 = timitExp.run_grids(grids_example)
