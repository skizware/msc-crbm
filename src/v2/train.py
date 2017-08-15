import datetime
import numpy as np
import os
import traceback

STATE_FILE_NAME='dbn_state{}.npy'

class DbnTrainer(object):
    def __init__(self, target_dbn, data_set, data_loader, stats_collector=None, output_state_path='./'):
        self.target_dbn = target_dbn
        self.data_set = data_set
        self.data_loader = data_loader
        self.stats_collector = stats_collector
        self.output_state_path = output_state_path

    def train_dbn(self, layer_idx, num_epochs=1, batch_size=1, state_saves_per_epoch=1):
        t_start = datetime.datetime.now()
        for epoch in xrange(0,num_epochs):
            print('Layer {} Epoch {} - Train Start: {}'.format(layer_idx, epoch, t_start))

            np.random.shuffle(self.data_set)

            batches_per_epoch = self.data_set.shape[0] / batch_size
            for batch in xrange(0, batches_per_epoch):
                start_index = batch * batch_size

                end_index = (batch + 1) * batch_size
                mini_batch_refs = self.data_set[start_index:end_index]
                mini_batch = self.data_loader.load_data(mini_batch_refs)
                if type(mini_batch) is list:
                    mini_batch = np.asarray(mini_batch)
                try:
                    stats = self.target_dbn.train_layer_on_batch(mini_batch)
                except Exception, e:
                    print "ERROR!!!"
                    print e
                    print "Caused By:"
                    print mini_batch_refs
                    print traceback.format_exc()

                if self.stats_collector is not None:
                    self.stats_collector.collect_stats(stats[0], stats[1], stats[2], stats[3], stats[4],
                          stats[5], stats[6], stats[7], stats[8], mini_batch, (epoch*batches_per_epoch) + batch, self.target_dbn, stats[9])

                if (batch+1) % int(batches_per_epoch/state_saves_per_epoch) is 0:
                    self.save_state('_' + str((epoch*batches_per_epoch) + (batch+1)))

            print('Layer {} Epoch {} - Train End - Elapsed time: {}'.format(layer_idx, epoch, datetime.datetime.now() - t_start))

    def save_state(self, substr=''):
        learned_state = self.target_dbn.get_learned_state()
        np.save(self.output_state_path + STATE_FILE_NAME.format(substr), learned_state)