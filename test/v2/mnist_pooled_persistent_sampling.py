import dbn
from experiment import MnistExperimentPersistentGibbs
from experiment import KEY_VIS_SHAPE, KEY_POOL_RATIO, KEY_HID_SHAPE, KEY_LAYER_TYPE, KEY_TARGET_SPARSITIES, KEY_LEARNING_RATES, KEY_SPARSITY_LEARNING_RATES, KEY_NUM_EPOCHS, KEY_BATCH_SIZE

DIR_OUT_RESULTS = 'mnist_persistent_batch_pooled/results/'
starting_dbn = dbn.BinaryVisibleDbnPersistentChainSampling()
mnistExp = MnistExperimentPersistentGibbs(starting_dbn, DIR_OUT_RESULTS)

grids_example = {
    KEY_VIS_SHAPE: (1, 1, 28, 28),
    KEY_HID_SHAPE: (1, 64, 18, 18),
    KEY_POOL_RATIO: 2,
    KEY_LEARNING_RATES: [0.1],
    KEY_TARGET_SPARSITIES: [0.1],
    KEY_SPARSITY_LEARNING_RATES: [0.9],
    KEY_BATCH_SIZE: 64,
    KEY_NUM_EPOCHS: 40
}

resultant2 = mnistExp.run_grids(grids_example)