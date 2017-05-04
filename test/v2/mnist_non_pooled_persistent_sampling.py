import dbn
from experiment import MnistExperiment
from experiment import KEY_VIS_SHAPE, KEY_POOL_RATIO, KEY_HID_SHAPE, KEY_LAYER_TYPE, KEY_TARGET_SPARSITIES, KEY_LEARNING_RATES, KEY_SPARSITY_LEARNING_RATES

DIR_OUT_RESULTS = 'mnist_persistent/results/'
starting_dbn = dbn.BinaryVisibleDbnPersistentChainSampling()
mnistExp = MnistExperiment(starting_dbn, DIR_OUT_RESULTS)

grids_example = {
    KEY_VIS_SHAPE: (1, 1, 28, 28),
    KEY_HID_SHAPE: (1, 32, 20, 20),
    KEY_POOL_RATIO: 0,
    KEY_LEARNING_RATES: [0.1],
    KEY_TARGET_SPARSITIES: [0.1],
    KEY_SPARSITY_LEARNING_RATES: [0.1]
}

resultant2 = mnistExp.run_grids(grids_example)