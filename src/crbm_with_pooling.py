import numpy as np
import theano as th
from theano import tensor as T

from pylearn2.expr.probabilistic_max_pooling import max_pool
from theano.tensor.shared_randomstreams import RandomStreams


class PooledCrbm(object):
    def __init__(self, numBases=3, visible_layer_shape=(1, 1, 18, 18), hidden_layer_shape=(1, 1, 16, 16),
                 sparsity_regularizarion_rate=0.003,
                 target_sparsity=0.003, learning_rate=0.001, pooling_ratio=2):
        self.rng = np.random.RandomState()
        self.theano_rng = RandomStreams(self.rng.randint(2 ** 30))
        self.visible_layer_shape = visible_layer_shape
        self.hidden_layer_shape = hidden_layer_shape
        self.numBases = numBases
        self.visible_bias = 0.
        self.hidden_group_biases = np.zeros(numBases, dtype=float)
        self.REGULARIZATION_RATE = sparsity_regularizarion_rate
        self.TARGET_SPARSITY = target_sparsity
        self.LEARNING_RATE = learning_rate
        self.pooling_ratio = pooling_ratio

        a = 1. / 100
        self.weight_groups = np.array(self.rng.uniform(  # initialize W uniformly
            low=-a,
            high=a,
            size=(self.__get_weight_matrix_shape())))

        self.__init_theano_vars()

        print self.weight_groups.shape

    def getStateObject(self):
        return {'hidden_layer_shape': self.get_hidden_layer_shape(), 'visible_layer_shape': self.get_vis_layer_shape(),
                'numBases': self.numBases, 'visible_bias': self.get_vis_bias(),
                'hidden_group_biases': self.get_hidden_group_biases(),
                'weight_groups': self.get_weight_groups(), 'REGULARIZATION_RATE': self.get_regularization_rate(),
                'TARGET_SPARSITY': self.get_target_sparsity(), 'LEARNING_RATE': self.get_learning_rate(),
                'type': type(self)}

    def loadStateObject(self, stateObject):
        self.set_hidden_layer_shape(stateObject['hidden_layer_shape'])
        self.set_vis_layer_shape(stateObject['visible_layer_shape'])
        self.numBases = stateObject['numBases']
        self.set_vis_bias(stateObject['visible_bias'])
        self.set_hidden_group_biases(stateObject['hidden_group_biases'])
        self.set_weight_groups(stateObject['weight_groups'])
        self.set_regularization_rate(stateObject['REGULARIZATION_RATE'])
        self.set_target_sparsity(stateObject['TARGET_SPARSITY'])
        self.set_learning_rate(stateObject['LEARNING_RATE'])

    def contrastive_divergence(self, trainingSample, previous_layer=None, previous_layer_visible_sample=None):

        h0_pre_sample, h0_sample, p, p_samples, p_bottom_up = self.sample_h_given_v(trainingSample)

        v1_pre_sample, v1_sample, \
        h1_pre_sample, h1_sample = self.gibbs_hvh(h0_sample, previous_layer, previous_layer_visible_sample)

        weight_group_delta = self.__th_update_weights(
            self.__get_hidden_layer_normalization_factor(),
            trainingSample, h0_pre_sample, v1_sample, h1_pre_sample)

        hidden_bias_delta, sparsity_delta, bias_updates = self.__th_update_hidden_biases(h0_pre_sample, h1_pre_sample)

        visible_bias_delta = self.__th_update_visible_bias(trainingSample, v1_sample)

        return [weight_group_delta, hidden_bias_delta, sparsity_delta, bias_updates, visible_bias_delta]

    def sample_h_given_v(self, inputMat):
        return self.__th_h_given_v(inputMat)

    def sample_h_given_p(self, p):
        return self.__th_h_given_p(p)

    def sample_h_given_p_none_h(self, p, none_h):
        return self.__th_h_given_p_and_none_h(p, none_h)

    def sample_v_given_h(self, hidden_groups):
        return self.__th_v_given_h(hidden_groups)

    def sample_p_given_v_and_top_down(self, v, top_down):
        return self.__th_p_given_v_and_top_down(v, top_down)

    def gibbs_vhv(self, inputMat):
        h_pre_sample, h_sample, p, p_sample, p_bottom_up = self.sample_h_given_v(inputMat)
        v_pre_sample, v_sample = self.sample_v_given_h(h_sample)
        return [v_pre_sample, v_sample, h_pre_sample, h_sample]

    def gibbs_hvh(self, hidden_groups, previous_layer=None, previous_layer_visible_sample=None):
        v_pre_sample, v_sample = self.sample_v_given_h(hidden_groups)

        if previous_layer_visible_sample is not None and previous_layer is not None:
            v_pre_sample, v_sample = previous_layer.sample_p_given_v_and_top_down(previous_layer_visible_sample, v_pre_sample)

        h_pre_sample, h_sample, p, p_sample, p_bottom_up = self.sample_h_given_v(v_sample)
        return [v_pre_sample, v_sample, h_pre_sample, h_sample]

    def __get_weight_matrix_shape(self):
        return (self.numBases, self.visible_layer_shape[1],) + \
               (self.visible_layer_shape[2] - self.hidden_layer_shape[2] + 1,
                self.visible_layer_shape[3] - self.hidden_layer_shape[3] + 1)

    def _sigmoid(self, x):
        return 1. / (1 + np.exp(-x))

    def get_vis_bias(self):
        return self.th_visible_bias.get_value()

    def set_vis_bias(self, vis_bias):
        self.th_visible_bias.set_value(vis_bias)

    def get_hidden_group_biases(self):
        return self.th_hidden_group_biases.get_value()

    def set_hidden_group_biases(self, hidden_group_biases):
        self.th_hidden_group_biases.set_value(hidden_group_biases)

    def get_weight_groups(self):
        return self.th_weight_groups.get_value()

    def set_weight_groups(self, weight_groups):
        self.th_weight_groups.set_value(weight_groups)

    def get_vis_layer_shape(self):
        return self.th_visible_layer_shape.get_value()

    def set_vis_layer_shape(self, vis_layer_shape):
        self.th_visible_layer_shape.set_value(vis_layer_shape)

    def get_hidden_layer_shape(self):
        return self.th_hidden_layer_shape.get_value()

    def set_hidden_layer_shape(self, hidden_layer_shape):
        self.th_hidden_layer_shape.set_value(hidden_layer_shape)

    def get_learning_rate(self):
        return self.th_learning_rate.get_value()

    def set_learning_rate(self, learning_rate):
        self.th_learning_rate.set_value(learning_rate)

    def get_regularization_rate(self):
        return self.th_regularization_rate.get_value()

    def set_regularization_rate(self, regularization_rate):
        self.th_regularization_rate.set_value(regularization_rate)

    def get_target_sparsity(self):
        return self.th_target_sparsity.get_value()

    def set_target_sparsity(self, target_sparsity):
        self.th_target_sparsity.set_value(target_sparsity)

    # =========================Theano Specifics=====================#
    def __init_theano_vars(self):
        self.th_visible_bias = th.shared(self.visible_bias, 'th_visible_bias')
        self.th_hidden_group_biases = th.shared(self.hidden_group_biases, 'th_hidden_group_biases')
        self.th_weight_groups = th.shared(self.weight_groups, 'th_weight_groups')
        self.th_visible_layer_shape = th.shared(self.visible_layer_shape, 'th_visible_layer_shape')
        self.th_hidden_layer_shape = th.shared(self.hidden_layer_shape, 'th_hidden_layer_shape')
        self.th_learning_rate = th.shared(self.LEARNING_RATE, 'th_learning_rate')
        self.th_regularization_rate = th.shared(self.REGULARIZATION_RATE, 'th_regularization_rate')
        self.th_target_sparsity = th.shared(self.TARGET_SPARSITY, 'th_target_sparsity')
        self.th_h_none = th.shared(np.zeros(self.get_hidden_layer_shape()))

        self.__th_h_given_v = self.__init__th_h_given_v()
        self.__th_v_given_h = self.__init__th_v_given_h()
        self.__th_p_given_v_and_top_down = self.__init__th_p_given_v_and_top_down()
        self.__th_update_weights = self.__init__th_update_weights()
        self.__th_update_hidden_biases = self.__init__th_update_hidden_biases()
        self.__th_update_visible_bias = self.__init__th_update_visible_bias()
        self.__th_h_given_p = self.__init__th_h_given_p()
        self.__th_h_given_p_and_none_h = self.__init__th_h_given_p_and_none_h()

    def __init__th_h_given_v(self):
        th_h_given_v_input = T.tensor4("th_h_given_v_input", dtype=th.config.floatX)
        th_h_given_v_bottom_up = T.nnet.conv2d(th_h_given_v_input, self.th_weight_groups, border_mode='valid',
                                                        filter_flip=False) + \
                                          self.th_hidden_group_biases.dimshuffle('x', 0, 'x', 'x')

        p, h, p_samples, h_samples = max_pool(th_h_given_v_bottom_up, (self.pooling_ratio, self.pooling_ratio), top_down=None,
                                              theano_rng=self.theano_rng)

        op = th.function(
            inputs=[th_h_given_v_input],
            outputs=[h, h_samples, p, p_samples, th_h_given_v_bottom_up]
        )

        return op

    def __init__th_p_given_v_and_top_down(self):
        th_p_given_v_input = T.tensor4("th_p_given_v_input", dtype=th.config.floatX)
        th_p_given_v_input_top_down = T.tensor4("th_h_given_v_input_top_down", dtype=th.config.floatX)
        th_p_given_v_bottom_up = T.nnet.conv2d(th_p_given_v_input, self.th_weight_groups, border_mode='valid',
                                               filter_flip=False) + \
                                 self.th_hidden_group_biases.dimshuffle('x', 0, 'x', 'x')

        p, h, p_samples, h_samples = max_pool(th_p_given_v_bottom_up, (self.pooling_ratio, self.pooling_ratio),
                                              top_down=th_p_given_v_input_top_down, theano_rng=self.theano_rng)

        op = th.function(
            inputs=[th_p_given_v_input, th_p_given_v_input_top_down],
            outputs=[p, p_samples]
        )

        return op

    def __init__th_h_given_p(self):
        th_h_given_p_input = T.tensor4("th_h_given_p_input", dtype=th.config.floatX)
        p, h, p_samples, h_samples = max_pool(self.th_h_none, (self.pooling_ratio, self.pooling_ratio),
                                              top_down=th_h_given_p_input, theano_rng=self.theano_rng)

        op = th.function(
            inputs=[th_h_given_p_input],
            outputs=[h, h_samples]
        )

        return op

    def __init__th_h_given_p_and_none_h(self):
        th_h_given_p_input = T.tensor4("th_h_given_p_input", dtype=th.config.floatX)
        th_h_given_p_none_h = T.tensor4("th_h_given_p_input", dtype=th.config.floatX)
        p, h, p_samples, h_samples = max_pool(th_h_given_p_none_h, (self.pooling_ratio, self.pooling_ratio),
                                              top_down=th_h_given_p_input, theano_rng=self.theano_rng)

        op = th.function(
            inputs=[th_h_given_p_input, th_h_given_p_none_h],
            outputs=[h, h_samples]
        )

        return op

    def __init__th_v_given_h(self):
        th_v_given_h_input = T.tensor4('th_v_given_h_input', dtype=th.config.floatX)

        th_v_given_h_output_pre_sample = T.nnet.conv2d(th_v_given_h_input, self.th_weight_groups.swapaxes(0, 1),
                                                       border_mode='full') + self.th_visible_bias
        th_v_given_h_output_sampled = self.theano_rng.normal(avg=th_v_given_h_output_pre_sample,
                                                             size=th_v_given_h_output_pre_sample.shape)

        op = th.function(
            inputs=[th_v_given_h_input],
            outputs=[th_v_given_h_output_pre_sample, th_v_given_h_output_sampled]
        )

        return op

    def __init__th_update_weights(self):
        th_input_sample = T.tensor4('th_inputSample', dtype=th.config.floatX)
        th_h0_pre_sample = T.tensor4('th_h0_pre_sample', dtype=th.config.floatX)
        th_v1_sampled = T.tensor4('th_v1_sampled', dtype=th.config.floatX)
        th_h1_pre_sample = T.tensor4('th_h1_pre_sample', dtype=th.config.floatX)
        th_normalization_factor = T.scalar('th_normalization_factor')

        def get_weight_updates_for_channel(channel_idx, vis_layer_shape, inputSample, h0_pre_sample, v1_sample,
                                           h1_pre_sample,
                                           learning_rate):
            return learning_rate * (1. / vis_layer_shape) * \
                   ((T.nnet.conv2d([inputSample[:, channel_idx, :, :]], h0_pre_sample.swapaxes(0, 1), filter_flip=False,
                                   border_mode='valid') - \
                     T.nnet.conv2d([v1_sample[:, channel_idx, :, :]], h1_pre_sample.swapaxes(0, 1), filter_flip=False,
                                   border_mode='valid'))[0])

        outputs, _ = th.scan(
            fn=get_weight_updates_for_channel,
            sequences=np.arange(self.th_visible_layer_shape.get_value()[1]),
            non_sequences=[th_normalization_factor, th_input_sample, th_h0_pre_sample, th_v1_sampled, th_h1_pre_sample,
                           self.th_learning_rate]
        )

        op = th.function(
            inputs=[th_normalization_factor, th_input_sample, th_h0_pre_sample, th_v1_sampled, th_h1_pre_sample],
            outputs=outputs.swapaxes(0, 1),
            updates=[(self.th_weight_groups, self.th_weight_groups + outputs.swapaxes(0, 1))]
        )

        return op

    def __init__th_update_hidden_biases(self):
        th_h0_pre_sample = T.tensor4('th_h0_pre_sample', dtype=th.config.floatX)
        th_h1_pre_sample = T.tensor4('th_h1_pre_sample', dtype=th.config.floatX)

        th_diff = th_h0_pre_sample - th_h1_pre_sample

        th_hidden_bias_delta = self.th_learning_rate * (1. / self.__get_hidden_layer_normalization_factor()) * (
            th_diff.sum(axis=(2, 3)))

        th_sparsity_delta = self.th_regularization_rate * (
            self.th_target_sparsity - (1. / self.__get_hidden_layer_normalization_factor()) * th_h0_pre_sample.sum(
                (0, 2, 3)))

        th_bias_updates = th_hidden_bias_delta + th_sparsity_delta

        op = th.function(
            inputs=[th_h0_pre_sample, th_h1_pre_sample],
            outputs=[th_hidden_bias_delta, th_sparsity_delta, th_bias_updates[0]],
            updates=[(self.th_hidden_group_biases, self.th_hidden_group_biases + th_bias_updates[0])]
        )

        return op

    def __get_hidden_layer_normalization_factor(self):
        return self.th_hidden_layer_shape.get_value()[2] * self.th_hidden_layer_shape.get_value()[3]

    def __init__th_update_visible_bias(self):
        th_v0 = T.tensor4('th_v0', dtype=th.config.floatX)
        th_v1 = T.tensor4('th_v1', dtype=th.config.floatX)

        th_diff = th_v0 - th_v1
        th_bias_update = self.th_learning_rate * (1. / self.__get_hidden_layer_normalization_factor()) * \
                         (th_diff.sum())

        op = th.function(
            inputs=[th_v0, th_v1],
            outputs=th_bias_update,
            updates=[(self.th_visible_bias, self.th_visible_bias + th_bias_update)]
        )

        return op


class PooledBinaryCrbm(PooledCrbm):
    def sample_v_given_h(self, hidden_groups):
        output_pre_sample, _ = super(PooledBinaryCrbm, self).sample_v_given_h(hidden_groups)
        output_sigmoid = self._sigmoid(output_pre_sample)
        output_sample = self.rng.binomial(n=1, p=output_sigmoid, size=output_sigmoid.shape)
        return [output_sigmoid, output_sample]
