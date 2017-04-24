from abc import ABCMeta, abstractmethod
from unit import LayerUnits
from inference import NonPooledInferenceBinaryVisible, NonPooledInferenceGaussianVisible, PooledInferenceBinaryVisible
from sampling import RbmGibbsSampler, RbmPersistentGibbsSampler
import numpy as np

import theano as th
from theano import tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import math

KEY_POOLING_RATIO = 'pooling_ratio'

KEY_LAYER_TYPE = 'layer_type'
KEY_HID_BIASES = 'hidden_biases'
KEY_VIS_BIAS = 'visible_bias'
KEY_WEIGHT_MATRIX = 'weight_matrix'
KEY_SPARSITY_LEARNING_RATE = 'sparsity_learning_rate'
KEY_TARGET_SPARSITY = 'target_sparsity'
KEY_LEARNING_RATE = 'learning_rate'
KEY_HID_SHAPE = 'hidden_shape'
KEY_VIS_SHAPE = 'visible_shape'


class AbstractLayer:
    __metaclass__ = ABCMeta

    """
    Abstract base class for a layer in a DBN.

    Layers have visible units, and hidden units.
    Visible Units have shape (batch size, num channels, in-rows, in-cols)
    Hidden units have shape (batch size, num learned bases, out-rows, out-cols)
    """

    def __init__(self, vis_unit_shape, hid_unit_shape, pre_set_vis_units=None, pre_set_hid_units=None,
                 learning_rate=0.01, target_sparsity=1., sparsity_learning_rate=0.):
        self.__init_units(hid_unit_shape, vis_unit_shape, pre_set_hid_units, pre_set_vis_units)
        self.__batch_size = vis_unit_shape[0]
        self.__num_bases = hid_unit_shape[1]
        self.__vis_bias = 0.
        self.__hid_biases = np.zeros((self.__num_bases,), dtype=float)
        self.__rng = np.random.RandomState()
        self.__learning_rate = learning_rate
        self.__hid_normalization_factor = hid_unit_shape[2] * hid_unit_shape[3]
        self.__vis_normalization_factor = vis_unit_shape[1] * vis_unit_shape[2] * vis_unit_shape[3]
        self.__target_sparsity = target_sparsity
        self.__sparsity_learning_rate = sparsity_learning_rate

        init_weight_variance = 1. / 100
        self.__weight_matrix = np.array(self.__rng.uniform(  # initialize W uniformly
            low=-init_weight_variance,
            high=init_weight_variance,
            size=(self.__get_weight_matrix_shape(vis_unit_shape, hid_unit_shape))))

        self.__th_setup()
        self.init_inference_proc()
        self.init_sampling_proc()

    def init_inference_proc(self):
        raise NotImplementedError("Do not use the abstract base class")

    def init_sampling_proc(self):
        raise NotImplementedError("Do not use the abstract base class")

    def train_on_minibatch(self, input_batch):
        pos_vis = input_batch
        pos_hid_infer, pos_hid_sampled = self.infer_hid_given_vis(input_batch)
        neg_vis_infer, neg_vis_sampled, neg_hid_infer, neg_hid_sampled = self.sample_from_model(pos_hid_sampled)

        weight_group_delta = self.__th_update_weights(pos_vis.swapaxes(0,1), pos_hid_infer.swapaxes(0,1), neg_vis_sampled.swapaxes(0,1), neg_hid_infer.swapaxes(0,1)).swapaxes(0,1)

        if math.isnan(np.average(weight_group_delta)):
            raise Exception("Some shit went NaN")

        if np.abs(np.average(weight_group_delta)) > 100:
            raise Exception("Hmm - fishy...")

        #print "Weight Delta Max: {}. Min: {}, Ave: {}".format(np.max(weight_group_delta),np.min(weight_group_delta), np.average(weight_group_delta))
        self.__th_weight_matrix.set_value(self.__th_weight_matrix.get_value() + weight_group_delta)
        # weight_group_delta = self.__th_update_weights(pos_vis, pos_hid_infer, neg_vis_sampled, neg_hid_infer)
        hidden_bias_delta, sparsity_delta, bias_updates = self.__th_update_hidden_biases(pos_hid_sampled,
                                                                                         neg_hid_sampled)
        #print "Hid Bias Delta Max: {}|{}. Min: {}|{}, Ave: {}|{}".format(np.max(hidden_bias_delta),np.max(sparsity_delta),np.min(hidden_bias_delta),np.min(sparsity_delta), np.average(hidden_bias_delta),np.average(sparsity_delta))

        visible_bias_delta = self.__th_update_visible_bias(pos_vis, neg_vis_sampled)
        #print "Vis Bias Delta Max: {}. Min: {}, Ave: {}".format(np.max(visible_bias_delta),np.min(visible_bias_delta), np.average(visible_bias_delta))

        recreation_squared_error = ((pos_vis - neg_vis_infer) ** 2).sum()

        return [weight_group_delta, hidden_bias_delta, sparsity_delta, bias_updates, visible_bias_delta, pos_hid_infer,
                neg_vis_infer,
                neg_hid_infer, recreation_squared_error, neg_vis_sampled]

    def get_num_hidden_groups(self):
        return self.__hid_units.get_shape()[1]

    def get_num_visible_channels(self):
        return self.__vis_units.get_shape()[1]

    def sample_from_model(self, initial_input=None):
        return self.sampling_proc.sample_from_distribution(initial_input)

    def infer_hid_given_vis(self, vis_input=None):
        if vis_input is None:
            vis_input = self.__vis_units.get_value()
            assert vis_input is not None, "Error - no value set for visible units"
        else:
            self.__vis_units.set_value(vis_input)
            self.__vis_units.set_expectation(None)

        ret = self.inference_proc.infer_hid_given_vis(vis_input)
        self.__hid_units.set_expectation(ret[0])
        self.__hid_units.set_value(ret[1])
        return ret

    def infer_vis_given_hid(self, hid_input=None):
        if hid_input is None:
            hid_input = self.__hid_units.get_value()
            assert hid_input is not None, "Error - no value set for hidden units"
        else:
            self.__hid_units.set_value(hid_input)
            self.__hid_units.set_expectation(None)

        ret = self.inference_proc.infer_vis_given_hid(hid_input)
        self.__vis_units.set_expectation(ret[0])
        self.__vis_units.set_value(ret[1])
        return ret

    def __init_units(self, hid_unit_shape, vis_unit_shape, pre_set_hid_units, pre_set_vis_units):
        if pre_set_vis_units is not None:
            assert vis_unit_shape == pre_set_vis_units.get_shape(), 'Unit shape mismatch with pre defined visible units'
            self.__vis_units = pre_set_vis_units
        else:
            self.__vis_units = LayerUnits(vis_unit_shape)
        if pre_set_hid_units is not None:
            assert hid_unit_shape == pre_set_hid_units.get_shape(), 'Unit shape mismatch with pre defined hidden units'
            self.__hid_units = pre_set_hid_units
        else:
            self.__hid_units = LayerUnits(hid_unit_shape)

        self.__vis_units.set_connected_up(self.__hid_units)
        self.__hid_units.set_connected_down(self.__vis_units)

    @staticmethod
    def __get_weight_matrix_shape(vis_unit_shape, hid_unit_shape):
        return (hid_unit_shape[1],
                vis_unit_shape[1],
                vis_unit_shape[2] - hid_unit_shape[2] + 1,
                vis_unit_shape[3] - hid_unit_shape[3] + 1)

    def get_weight_matrix(self):
        return self.__th_weight_matrix

    def get_hidden_biases(self):
        return self.__th_hid_biases

    def get_hidden_units(self):
        return self.__hid_units

    def get_visible_units(self):
        return self.__vis_units

    def get_visible_bias(self):
        return self.__th_vis_bias

    def get_rng(self):
        return self.__theano_rng

    def get_learning_rate(self):
        return self.__learning_rate

    def get_target_sparsity(self):
        return self.__target_sparsity

    def get_sparsity_learning_rate(self):
        return self.__sparsity_learning_rate

    def get_state_object(self):
        state = {KEY_VIS_SHAPE: self.__vis_units.get_shape(), KEY_HID_SHAPE: self.__hid_units.get_shape(),
                 KEY_LEARNING_RATE: self.__learning_rate, KEY_TARGET_SPARSITY: self.__target_sparsity,
                 KEY_SPARSITY_LEARNING_RATE: self.__sparsity_learning_rate,
                 KEY_WEIGHT_MATRIX: self.__th_weight_matrix.get_value(),
                 KEY_VIS_BIAS: self.__th_vis_bias.get_value(),
                 KEY_HID_BIASES: self.__th_hid_biases.get_value(),
                 KEY_LAYER_TYPE: type(self)}

        return state

    def set_weight_matrix(self, weight_matrix):
        self.__th_weight_matrix.set_value(weight_matrix)
        self.__weight_matrix = weight_matrix

    def set_vis_bias(self, vis_bias):
        self.__th_vis_bias.set_value(vis_bias)
        self.__vis_bias = vis_bias

    def set_hid_bias(self, hid_biases):
        self.__th_hid_biases.set_value(hid_biases)
        self.__hid_biases = hid_biases

    def set_learning_rate(self, learning_rate):
        self.__th_learning_rate.set_value(learning_rate)
        self.__learning_rate = learning_rate

    def set_target_sparsity(self, target_sparsity):
        self.__th_target_sparsity.set_value(target_sparsity)
        self.__target_sparsity = target_sparsity

    def set_sparsity_learning_rate(self, sparsity_learning_rate):
        self.__th_sparsity_learning_rate.set_value(sparsity_learning_rate)
        self.__sparsity_learning_rate = sparsity_learning_rate

    def set_internal_state(self, learned_state):
        self.set_weight_matrix(learned_state[KEY_WEIGHT_MATRIX])
        self.set_hid_bias(learned_state[KEY_HID_BIASES])
        self.set_vis_bias(learned_state[KEY_VIS_BIAS])

    # ==============================Theano Specifics================================== #
    def __th_setup(self):
        self.__theano_rng = RandomStreams(self.__rng.randint(2 ** 30))
        self.th_var_init()
        self.__th_op_init()

    def __th_op_init(self):
        self.__th_update_weights = self.__init__th_update_weights()
        self.__th_update_hidden_biases = self.__init__th_update_hidden_biases()
        self.__th_update_visible_bias = self.__init__th_update_visible_bias()

    def th_var_init(self):
        self.__th_vis_bias = th.shared(self.__vis_bias)
        self.__th_hid_biases = th.shared(self.__hid_biases)
        self.__th_weight_matrix = th.shared(self.__weight_matrix)
        self.__th_vis_shape = th.shared(self.__vis_units.get_shape())
        self.__th_hid_shape = th.shared(self.__hid_units.get_shape())
        self.__th_learning_rate = th.shared(self.__learning_rate)
        self.__th_hid_normalization_factor = th.shared(self.__hid_normalization_factor)
        self.__th_vis_normalization_factor = th.shared(self.__vis_normalization_factor)
        self.__th_target_sparsity = th.shared(self.__target_sparsity)
        self.__th_sparsity_learning_rate = th.shared(self.__sparsity_learning_rate)

    def __init__th_update_weights(self):
        """NB - Assumes the input for visible samples is (NumVisChannels, 1, R, C) and input for hidden samples is 
        (NumHidChannels, 1, R, C) """
        th_input_sample = T.tensor4('th_inputSample', dtype=th.config.floatX)
        th_h0_pre_sample = T.tensor4('th_h0_pre_sample', dtype=th.config.floatX)
        th_v1_sampled = T.tensor4('th_v1_sampled', dtype=th.config.floatX)
        th_h1_pre_sample = T.tensor4('th_h1_pre_sample', dtype=th.config.floatX)

        output = self.__th_learning_rate * (1. / self.__th_hid_normalization_factor) * \
                 (T.nnet.conv2d(th_input_sample, th_h0_pre_sample, filter_flip=False, border_mode='valid') -
                  T.nnet.conv2d(th_v1_sampled, th_h1_pre_sample, filter_flip=False, border_mode='valid'))

        op = th.function(
            inputs=[th_input_sample, th_h0_pre_sample, th_v1_sampled, th_h1_pre_sample],
            outputs=output
        )

        return op

    def __init__th_update_hidden_biases(self):
        th_h0_pre_sample = T.tensor4('th_h0_pre_sample', dtype=th.config.floatX)
        th_h1_pre_sample = T.tensor4('th_h1_pre_sample', dtype=th.config.floatX)

        th_diff = th_h0_pre_sample - th_h1_pre_sample

        th_hidden_bias_delta = self.__th_learning_rate * (1. / self.__th_hid_normalization_factor) * (
            th_diff.sum(axis=(0, 2, 3)))

        th_sparsity_delta = self.__th_sparsity_learning_rate * (
            self.__th_target_sparsity - (1. / self.__th_hid_normalization_factor) * th_h0_pre_sample.sum(axis=(0, 2, 3)))

        th_bias_updates = th_hidden_bias_delta + th_sparsity_delta

        op = th.function(
            inputs=[th_h0_pre_sample, th_h1_pre_sample],
            outputs=[th_hidden_bias_delta, th_sparsity_delta, th_bias_updates],
            updates=[(self.__th_hid_biases, self.__th_hid_biases + th_bias_updates)]
        )

        return op

    def __init__th_update_visible_bias(self):
        th_v0 = T.tensor4('th_v0', dtype=th.config.floatX)
        th_v1 = T.tensor4('th_v1', dtype=th.config.floatX)

        th_diff = th_v0 - th_v1
        th_bias_update = self.__th_learning_rate * (1. / self.__th_vis_normalization_factor) * \
                         (th_diff.sum())

        op = th.function(
            inputs=[th_v0, th_v1],
            outputs=th_bias_update,
            updates=[(self.__th_vis_bias, self.__th_vis_bias + th_bias_update)]
        )

        return op


class BinaryVisibleNonPooledLayer(AbstractLayer):
    def __init__(self, vis_unit_shape, hid_unit_shape, pre_set_vis_units=None, pre_set_hid_units=None,
                 learning_rate=0.01, target_sparsity=0.01, sparsity_learning_rate=0.9):
        super(BinaryVisibleNonPooledLayer, self).__init__(vis_unit_shape, hid_unit_shape, pre_set_vis_units,
                                                          pre_set_hid_units, learning_rate, target_sparsity,
                                                          sparsity_learning_rate)

    def init_inference_proc(self):
        self.inference_proc = NonPooledInferenceBinaryVisible(self)

    def init_sampling_proc(self):
        self.sampling_proc = RbmGibbsSampler(self)


class BinaryVisiblePooledLayer(AbstractLayer):
    def __init__(self, vis_unit_shape, hid_unit_shape, pre_set_vis_units=None, pre_set_hid_units=None,
                 learning_rate=0.01, target_sparsity=1., sparsity_learning_rate=0., pooling_ratio=2, pre_set_pooling_units=None):
        self.__pooling_ratio = pooling_ratio
        super(BinaryVisiblePooledLayer, self).__init__(vis_unit_shape, hid_unit_shape, pre_set_vis_units,
                                                       pre_set_hid_units, learning_rate, target_sparsity,
                                                       sparsity_learning_rate)

        self.__init_pool_units(hid_unit_shape, pre_set_pooling_units)

    def __init_pool_units(self, hid_unit_shape, pre_set_pooling_units):
        if pre_set_pooling_units is None:
            assert hid_unit_shape[2] % self.__pooling_ratio == 0 & \
                   hid_unit_shape[3] % self.__pooling_ratio == 0, \
                   "Pooling layer shape not compatible with pooling ratio"

            pooling_layer_shape = (hid_unit_shape[0], hid_unit_shape[1], hid_unit_shape[2] / self.__pooling_ratio,
                                   hid_unit_shape[3] / self.__pooling_ratio)
            self.__pool_units = LayerUnits(pooling_layer_shape)
        else:
            assert hid_unit_shape[2] % pre_set_pooling_units.get_shape()[2] == 0 & \
                   hid_unit_shape[3] % pre_set_pooling_units.get_shape()[3] == 0 & \
                   hid_unit_shape[2] / pre_set_pooling_units.get_shape()[2] == hid_unit_shape[3] / pre_set_pooling_units.get_shape()[3],\
                   "Pooling setup fail"

            self.__pool_units = pre_set_pooling_units

        self.get_hidden_units().set_connected_up(self.__pool_units)
        self.__pool_units.set_connected_down(self.get_hidden_units())

    def init_inference_proc(self):
        self.inference_proc = PooledInferenceBinaryVisible(self)

    def init_sampling_proc(self):
        self.sampling_proc = RbmGibbsSampler(self)

    def get_pool_ratio(self):
        return tuple((self.__pooling_ratio, self.__pooling_ratio))

    def get_pool_units(self):
        return self.__pool_units

    def get_state_object(self):
        state = super(BinaryVisiblePooledLayer, self).get_state_object()
        state[KEY_POOLING_RATIO] = self.__pooling_ratio

    def set_internal_state(self, learned_state):
        super(BinaryVisiblePooledLayer, self).set_internal_state(learned_state)
        self.__pooling_ratio = learned_state[KEY_POOLING_RATIO]


class BinaryVisibleNonPooledPersistentSamplerChainLayer(BinaryVisibleNonPooledLayer):
    def init_sampling_proc(self):
        self.sampling_proc = RbmPersistentGibbsSampler(self)


class GaussianVisibleNonPooledLayer(AbstractLayer):
    def __init__(self, vis_unit_shape, hid_unit_shape, pre_set_vis_units=None, pre_set_hid_units=None,
                 learning_rate=0.01, target_sparsity=0.01, sparsity_learning_rate=0.9):
        super(GaussianVisibleNonPooledLayer, self).__init__(vis_unit_shape, hid_unit_shape, pre_set_vis_units,
                                                            pre_set_hid_units, learning_rate, target_sparsity,
                                                            sparsity_learning_rate)

    def init_inference_proc(self):
        self.inference_proc = NonPooledInferenceGaussianVisible(self)

    def init_sampling_proc(self):
        self.sampling_proc = RbmGibbsSampler(self)
