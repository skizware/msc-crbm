import dbn
import numpy as np
from experiment import MnistExperiment
from experiment import KEY_VIS_SHAPE, KEY_POOL_RATIO, KEY_HID_SHAPE, KEY_RE_TRAIN_TOP_LAYER, KEY_TARGET_SPARSITIES, KEY_LEARNING_RATES, KEY_SPARSITY_LEARNING_RATES, KEY_BATCH_SIZE, KEY_NUM_EPOCHS

DIR_OUT_RESULTS = 'results/mnist2/'
starting_dbn = dbn.BinaryVisibleDbn()
#starting_dbn = dbn.DbnFromStateBuilder.init_dbn(np.load('/home/dave/code/msc-crbm/test/v2/mnist_batch_64_non_pooled_cd1/results/layer_0/lr_0.1_st_0.04_slr_0.9/dbn_state.npy').item())
mnistExp = MnistExperiment(starting_dbn, DIR_OUT_RESULTS)

grids_example = {
    KEY_VIS_SHAPE: (1, 1, 28, 28),
    KEY_HID_SHAPE: (1, 64, 22, 22),
    KEY_POOL_RATIO: None,
    KEY_LEARNING_RATES: [0.1],
    KEY_TARGET_SPARSITIES: [0.05],
    KEY_SPARSITY_LEARNING_RATES: [0.9],
    KEY_BATCH_SIZE: 16,
    KEY_NUM_EPOCHS: 200,
    KEY_RE_TRAIN_TOP_LAYER: False
}

resultant2 = mnistExp.run_grids(grids_example)