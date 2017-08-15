import dbn
from experiment import TimitExperiment
from sklearn.linear_model import SGDClassifier
from sklearn.utils import shuffle
from math import ceil, floor
import numpy as np


class SGDClassificationDbnVerifier(object):
    def __init__(self, dbn_to_verify, train_set_refs, test_set_refs, class_labels, data_loader, num_train_epochs=5,
                 train_batch_size=None, test_batch_size=None):
        super(SGDClassificationDbnVerifier, self).__init__()
        self.dbn_to_verify = dbn_to_verify
        self.train_set_refs = train_set_refs
        self.test_set_refs = test_set_refs
        self.class_labels = class_labels
        self.data_loader = data_loader
        self.num_train_epochs = num_train_epochs
        self.linear_model = SGDClassifier()
        self.train_batch_size = len(self.train_set_refs[0]) if train_batch_size is None else train_batch_size
        self.test_batch_size = len(self.test_set_refs[0]) if test_batch_size is None else test_batch_size

    def verify_model(self, dbn_layers_to_use=None):
        resulting_points = []
        for epoch in xrange(0, self.num_train_epochs):
            print "EPOCH {}".format(epoch)
            self.__train_classifier(dbn_layers_to_use)
            resulting_points.append(self.__test_classifier(dbn_layers_to_use))
            shuffle(self.train_set_refs[0], self.train_set_refs[1])

        return resulting_points

    def __train_classifier(self, dbn_layers_to_use):
        # decide on the number of batches we need to run thru for an epoch
        num_batches = int(ceil(len(self.train_set_refs[0]) / float(self.train_batch_size)))
        for batch_num in xrange(0, num_batches):
            print "BATCH {} - TRAIN".format(batch_num)
            raw_batch_data, raw_batch_labels = self.__get_raw_batch_data_and_labels(batch_num, self.train_set_refs,
                                                                                    self.train_batch_size)
            assert len(raw_batch_data) == len(raw_batch_labels), "Error, mis-match on sample num vs label num"

            self.__train_model_on_batch(dbn_layers_to_use, raw_batch_data, raw_batch_labels)

    def __train_model_on_batch(self, dbn_layers_to_use, raw_batch_data, raw_batch_labels):
        labeled_data_points = self.__labeled_data_points_from_raw_batch(raw_batch_data, raw_batch_labels)
        batch_data, batch_labels = self.__chunk_batch_data_and_labels(labeled_data_points)
        shuffle(batch_data, batch_labels)
        batch_features = self.dbn_to_verify.get_features(np.asarray(batch_data), dbn_layers_to_use)
        self.linear_model.partial_fit(batch_features, np.asarray(batch_labels), self.class_labels)

    @staticmethod
    def __chunk_batch_data_and_labels(labeled_data_points):
        batch_data = []
        batch_labels = []
        for data_point in labeled_data_points:
            chunked_batch_data, chunked_batch_labels = data_point.get_subsamples_and_labels()
            batch_data += chunked_batch_data
            batch_labels += chunked_batch_labels
        return batch_data, batch_labels

    def __labeled_data_points_from_raw_batch(self, raw_batch_data, raw_batch_labels):
        labeled_data_points = []
        for sample_count in xrange(0, len(raw_batch_data)):
            labeled_data_sample = self.get_labeled_data_sample(raw_batch_data[sample_count:sample_count + 1][0],
                                                                 raw_batch_labels[sample_count:sample_count + 1][0])
            labeled_data_points.append(labeled_data_sample)
        return labeled_data_points

    def __get_raw_batch_data_and_labels(self, batch_num, example_set_refs, batch_size):
        batch_data_refs = example_set_refs[0][batch_num * batch_size:(batch_num + 1) * batch_size]
        batch_label_refs = example_set_refs[1][batch_num * batch_size:(batch_num + 1) * batch_size]

        raw_batch_data = self.data_loader.load_data(batch_data_refs)
        raw_batch_labels = self.data_loader.load_labels(batch_label_refs)

        return raw_batch_data, raw_batch_labels

    def get_labeled_data_sample(self, sample_data, sample_label):
        return LabeledDataSample(sample_data, sample_label)

    def __test_classifier(self, dbn_layers_to_use):
        predicted_data_points = []
        num_batches = int(ceil(len(self.test_set_refs[0]) / float(self.test_batch_size)))
        for batch_num in xrange(0, num_batches):
            print "BATCH {} - TEST".format(batch_num)
            raw_batch_data, raw_batch_labels = self.__get_raw_batch_data_and_labels(batch_num, self.test_set_refs,
                                                                                    self.test_batch_size)
            labeled_data_points = self.__labeled_data_points_from_raw_batch(raw_batch_data, raw_batch_labels)

            for data_point in labeled_data_points:
                subsamples, labels = data_point.get_subsamples_and_labels()
                features = self.dbn_to_verify.get_features(subsamples, dbn_layers_to_use)
                predictions = self.linear_model.predict(features)
                unique, counts = np.unique(predictions, return_counts=True)
                counted_results = dict(zip(unique, counts))
                most_predicted = max(counted_results, key=lambda key: counted_results[key])
                data_point.predicted_label = most_predicted
                predicted_data_points.append(data_point)

        return predicted_data_points


class TimitSGDClassificationDbnVerifier(SGDClassificationDbnVerifier):
    def __init__(self, dbn_to_verify, train_set_refs, test_set_refs, class_labels, data_loader, time_bins_per_subsample,
                 num_train_epochs=5, train_batch_size=None, test_batch_size=None):
        super(TimitSGDClassificationDbnVerifier, self).__init__(dbn_to_verify, train_set_refs, test_set_refs,
                                                                class_labels, data_loader, num_train_epochs,
                                                                train_batch_size, test_batch_size)

        self.time_bins_per_subsample = time_bins_per_subsample

    def get_labeled_data_sample(self, sample_data, sample_label):
        return LabeledTimitDataSample(sample_data, sample_label, time_bins_per_subsample=self.time_bins_per_subsample)


class LabeledDataSample(object):
    def __init__(self, raw_data, class_label):
        self.class_label = class_label
        self.__subsamples = self.prepare_subsamples(raw_data)
        self.predicted_label = None

    def prepare_subsamples(self, raw_data):
        return [raw_data]

    def __get_subsample_labels(self):
        return [self.class_label] * len(self.__subsamples)

    def get_subsamples_and_labels(self):
        return [self.__subsamples, self.__get_subsample_labels()]


class LabeledTimitDataSample(LabeledDataSample):
    def __init__(self, raw_data, class_label, time_bins_per_subsample):
        self.time_bins_per_subsample = time_bins_per_subsample
        super(LabeledTimitDataSample, self).__init__(raw_data, class_label)

    def prepare_subsamples(self, raw_data):
        """raw_data is in format (1, numFreqBins, 1, numTimeBins)"""
        subsamples = []
        total_time_bins = raw_data.shape[2]
        total_sub_samples = int(floor(total_time_bins / float(self.time_bins_per_subsample)))
        for count in xrange(0, total_sub_samples):
            subsamples.append(
                raw_data[:, :, count * self.time_bins_per_subsample:(count + 1) * self.time_bins_per_subsample]
            )

        return subsamples
