import rbm_trainer
import numpy as np
import os


class DbnTrainer(object):
    def __init__(self, dbn, train_set, valid_set, stats_collection_period=500, num_training_epochs=1,
                 output_directory='', image_loader=None):
        self.rbm_trainers = []
        self.target_dbn = dbn
        self.train_set = train_set
        self.valid_set = valid_set
        count = 0
        for layer in self.target_dbn.layers:
            trainer = rbm_trainer.RbmTrainer(layer, train_set, valid_set, stats_collection_period, num_training_epochs,
                                             output_directory + 'layer_{}/'.format(count), image_loader)

            self.rbm_trainers.append(trainer)
            count += 1

    def train_dbn_unsupervised(self, starting_layer=0, ending_layer=-1):
        if ending_layer == -1:
            ending_layer = len(self.target_dbn.layers) - 1

        for layer_idx in range(starting_layer, ending_layer+1):
            print "TRAINING LAYER {}".format(layer_idx)
            trainer = self.rbm_trainers[layer_idx]
            recreation_err_squared = []
            np.random.shuffle(self.train_set)
            iteration_count = 0
            for image in self.train_set:
                sample = image.copy()
                if layer_idx != 0:
                    sample = self.target_dbn.sample_h_given_v(image, ending_layer=layer_idx - 1)[1]
                hidden_bias_delta, sparsity_delta, \
                visible_bias_delta, weight_group_delta = trainer.train_given_sample(sample)

                if (iteration_count % 500 == 0):
                    test_idx = np.random.randint(1, len(self.valid_set) - 1)
                    test_sample = self.valid_set[test_idx].copy()
                    recreation = self.target_dbn.recreation_at_layer(test_sample, layer_idx)
                    trainer.collect_statistics(hidden_bias_delta, iteration_count, recreation_err_squared,
                                               sparsity_delta, weight_group_delta, 1, test_sample, recreation)

                iteration_count += 1

