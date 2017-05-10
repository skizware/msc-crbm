import dbn
from experiment import CaltechExperimentPersistentGibbs
from experiment import KEY_VIS_SHAPE, KEY_POOL_RATIO, KEY_HID_SHAPE, KEY_LAYER_TYPE, KEY_TARGET_SPARSITIES, KEY_LEARNING_RATES, KEY_SPARSITY_LEARNING_RATES

DIR_OUT_RESULTS = 'caltech_persistent_pooled/results/'
starting_dbn = dbn.GaussianVisibleDbnPersistentChainSampling()
caltechExp = CaltechExperimentPersistentGibbs(starting_dbn, DIR_OUT_RESULTS)

grids_example = {
    KEY_VIS_SHAPE: (1, 1, 200, 300),
    KEY_HID_SHAPE: (1, 64, 192, 292),
    KEY_POOL_RATIO: 2,
    KEY_LEARNING_RATES: [0.001],
    KEY_TARGET_SPARSITIES: [0.1, 0.3],
    KEY_SPARSITY_LEARNING_RATES: [0.1]
}

resultant = caltechExp.run_grids(grids_example)

grids_example = {
    KEY_VIS_SHAPE: (1, 1, 200, 300),
    KEY_HID_SHAPE: (1, 64, 192, 292),
    KEY_POOL_RATIO: 2,
    KEY_LEARNING_RATES: [0.001],
    KEY_TARGET_SPARSITIES: [1.],
    KEY_SPARSITY_LEARNING_RATES: [0.]
}

resultant2 = caltechExp.run_grids(grids_example)