import datetime
import numpy as np
import os

STATE_FILE_NAME='dbn_state{}.npy'

class DbnTrainer(object):
    def __init__(self, target_dbn, data_set, data_loader, stats_collector=None, output_state_path='./'):
        self.target_dbn = target_dbn
        self.data_set = data_set
        self.data_loader = data_loader
        self.stats_collector = stats_collector
        self.output_state_path = output_state_path

    def train_dbn(self, layer_idx):
        t_start = datetime.datetime.now()
        print('Layer {} - Train Start: {}'.format(layer_idx, t_start))
        np.random.shuffle(self.data_set)

        sample_index = 0
        for data_ref in self.data_set:
            train_input = self.data_loader.load_data(data_ref)
            stats = self.target_dbn.train_layer_on_batch(train_input)

            if self.stats_collector is not None:
                self.stats_collector.collect_stats(stats[0], stats[1], stats[2], stats[3], stats[4],
                      stats[5], stats[6], stats[7], stats[8], train_input, sample_index, self.target_dbn)

            if sample_index % 5000 is 0:
                self.save_state('_' + str(sample_index))

            sample_index += 1

        print('Layer {} - Train End - Elapsed time: {}'.format(layer_idx, datetime.datetime.now() - t_start))

    def save_state(self, substr=''):
        learned_state = self.target_dbn.get_learned_state()
        np.save(self.output_state_path + STATE_FILE_NAME.format(substr), learned_state)