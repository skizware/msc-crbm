import datetime
import numpy as np


class DbnTrainer(object):
    def __init__(self, target_dbn, data_set, data_loader, stats_collector=None):
        self.target_dbn = target_dbn
        self.data_set = data_set
        self.data_loader = data_loader
        self.stats_collector = stats_collector

    def train_dbn(self, layer_idx):
        t_start = datetime.datetime.now()
        print('Layer {} - Train Start: {}'.format(layer_idx, t_start))
        np.random.shuffle(self.data_set)

        sample_index = 0
        for data_ref in self.data_set:
            train_input = self.data_loader.load_data(data_ref)
            stats = self.target_dbn.train_layer_on_batch(train_input)

            if self.stats_collector is not None:
                self.stats_collector.collect_stats(stats, sample_index)

            sample_index += 1

        print('Layer {} - Train End - Elapsed time: {}'.format(layer_idx, datetime.datetime.now() - t_start))