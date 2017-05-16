import dbn
from experiment import MnistExperiment
from experiment import KEY_VIS_SHAPE, KEY_POOL_RATIO, KEY_HID_SHAPE, KEY_LAYER_TYPE, KEY_TARGET_SPARSITIES, KEY_LEARNING_RATES, KEY_SPARSITY_LEARNING_RATES, KEY_BATCH_SIZE, KEY_NUM_EPOCHS

DIR_OUT_RESULTS = 'mnist_batch_128_non_pooled_cd1/results/'
starting_dbn = dbn.BinaryVisibleDbn()
mnistExp = MnistExperiment(starting_dbn, DIR_OUT_RESULTS)

grids_example = {
    KEY_VIS_SHAPE: (1, 1, 28, 28),
    KEY_HID_SHAPE: (1, 64, 20, 20),
    KEY_POOL_RATIO: 0,
    KEY_LEARNING_RATES: [0.1, 0.01],
    KEY_TARGET_SPARSITIES: [0.1, 0.3],
    KEY_SPARSITY_LEARNING_RATES: [0.1],
    KEY_BATCH_SIZE: 128,
    KEY_NUM_EPOCHS: 40
}

resultant2 = mnistExp.run_grids(grids_example)

grids_example = {
    KEY_VIS_SHAPE: (1, 1, 28, 28),
    KEY_HID_SHAPE: (1, 64, 20, 20),
    KEY_POOL_RATIO: 0,
    KEY_LEARNING_RATES: [0.1, 0.01],
    KEY_TARGET_SPARSITIES: [1],
    KEY_SPARSITY_LEARNING_RATES: [0],
    KEY_BATCH_SIZE: 128,
    KEY_NUM_EPOCHS: 40
}

resultant2 = mnistExp.run_grids(grids_example)