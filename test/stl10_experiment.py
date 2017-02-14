import os
import numpy as np
import stl10

import crbm

from rbm_trainer import RbmTrainer


def run_experiment():
    for lr in learning_rates:
        for ts in target_sparsities:
            for sc in sparsity_constants:
                np.random.shuffle(train_set)
                print("Running Experiment For Vals {} {} {} - START".format(lr, ts, sc))
                if not os.path.isdir(get_subdir_name(lr, sc, ts)):
                    os.makedirs(get_subdir_name(lr, sc, ts) + '/recreations')
                    os.makedirs(get_subdir_name(lr, sc, ts) + '/histograms')

                    myRbm = crbm.crbm(number_of_bases, visible_layer_shape, hidden_layer_shape, sc, ts, lr)
                    trainer = RbmTrainer(myRbm, train_set, valid_set, output_directory='stl10/test/', num_training_epochs=3)
                    try:
                        recreation_err_sqrd = []
                        iteration_count = 0
                        for image in train_set:
                            input_image = image.copy()
                            input_image = trainer.normalize_image(image)
                            hidden_bias_delta, sparsity_delta, \
                            visible_bias_delta, weight_group_delta = trainer.train_given_sample(input_image)

                            if(iteration_count % 500 == 0):
                                recreation = myRbm.gibbs_vhv(input_image)
                                trainer.collect_statistics(hidden_bias_delta, iteration_count, recreation_err_sqrd,
                                                           sparsity_delta, weight_group_delta, 1, image, recreation)

                            iteration_count += 1
                    except:
                        pass
                print("Running Experiment For Vals {} {} {} - END".format(lr, ts, sc))


def get_subdir_name(lr, sc, ts):
    return 'stl10/test/' + str(lr) + '_' + str(ts) + '_' + str(sc)


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


learning_rates = [0.01]
target_sparsities = [0.09, 0.07]
sparsity_constants = [0.95]

number_of_bases = 36
visible_layer_shape = (1, 1, 96, 96)
hidden_layer_shape = (1, 1, 90, 90)

print("Loading Training Set - START")
train_set = stl10.read_all_images(stl10.DATA_PATH)
print("Loading Training Set - END")

print("Initializing Training Set - START")
train_set = train_set.transpose(0, 3, 2, 1)
train_set_grayscale = rgb2gray(train_set)
train_set_grayscale = train_set_grayscale.reshape(5000, 1, 1, 96, 96)
train_set = train_set_grayscale[:4500, :, :, :, :]
valid_set = train_set_grayscale[4500:, :, :, :, :]
print("Initializing Training Set - END")

run_experiment()
