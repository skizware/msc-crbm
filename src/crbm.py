import numpy as np
import theano as th
from theano import tensor as T

from theano.tensor.shared_randomstreams import RandomStreams

LEARNING_RATE = 0.002


class crbm(object):
    def __init__(self, numBases=3, visible_layer_shape=(1, 1, 18, 18), hidden_layer_shape=(1, 1, 16, 16)):
        self.rng = np.random.RandomState(1234)
        self.theano_rng = RandomStreams(self.rng.randint(2 ** 30))
        self.visible_layer_shape = visible_layer_shape
        self.hidden_layer_shape = hidden_layer_shape
        self.numBases = numBases
        self.visible_bias = 0.
        self.hidden_group_biases = np.zeros(numBases, dtype=float)

        a = 1. / 10000
        self.weight_groups = np.array(self.rng.uniform(  # initialize W uniformly
            low=-a,
            high=a,
            size=(self.__get_weight_matrix_shape())))

        self.__init_theano_vars()

        print self.weight_groups.shape

    def getStateObject(self):
        return {'hidden_layer_shape': self.hidden_layer_shape, 'visible_layer_shape': self.visible_layer_shape, \
                'numBases': self.numBases, 'visible_bias': self.visible_bias,
                'hidden_group_biases': self.hidden_group_biases, \
                'weight_groups': self.weight_groups}

    def loadStateObject(self, stateObject):
        self.hidden_layer_shape = stateObject['hidden_layer_shape']
        self.visible_layer_shape = stateObject['visible_layer_shape']
        self.numBases = stateObject['numBases']
        self.visible_bias = stateObject['visible_bias']
        self.hidden_group_biases = stateObject['hidden_group_biases']
        self.weight_groups = stateObject['weight_groups']

    def contrastive_divergence(self, trainingSample):
        h0_pre_sample, h0_sample = self.sample_h_given_v(trainingSample)
        v1_pre_sample, v1_sample, \
        h1_pre_sample, h1_sample = self.gibbs_hvh(h0_sample)

        self.__th_update_weights(trainingSample, h0_pre_sample, v1_sample, h1_pre_sample, LEARNING_RATE)

        self.__th_update_hidden_biases(h0_pre_sample, h1_pre_sample, LEARNING_RATE)

        self.__th_update_visible_bias(trainingSample, v1_sample, LEARNING_RATE)

        # self.hidden_group_biases[group] += (0.003 - (1./self.hidden_layer_shape[0]**2)*(np.nansum(h0_pre_sample[group])))

        print "nansum trainingsample = " + str(np.nansum(trainingSample))
        print "nansum v1_sample = " + str(np.nansum(v1_sample))
        # self.hidden_group_biases = self.hidden_group_biases - 0.0001

    def sample_h_given_v(self, inputMat):
        return self.__th_h_given_v(inputMat)

    def sample_v_given_h(self, hidden_groups):
        return self.__th_v_given_h(hidden_groups)

    def gibbs_vhv(self, inputMat):
        h_pre_sample, h_sample = self.sample_h_given_v(inputMat)
        v_pre_sample, v_sample = self.sample_v_given_h(h_sample)
        return [v_pre_sample, v_sample, h_pre_sample, h_sample]

    def gibbs_hvh(self, hidden_groups):
        v_pre_sample, v_sample = self.sample_v_given_h(hidden_groups)
        h_pre_sample, h_sample = self.sample_h_given_v(v_sample)
        return [v_pre_sample, v_sample, h_pre_sample, h_sample]

    def __get_weight_matrix_shape(self):
        return (self.numBases, self.visible_layer_shape[1],) + \
               (self.visible_layer_shape[2] - self.hidden_layer_shape[2] + 1,
                self.visible_layer_shape[3] - self.hidden_layer_shape[3] + 1)

    def __sigmoid(self, x):
        return 1. / (1 + np.exp(-x))

    # =========================Theano Specifics=====================#
    def __init_theano_vars(self):
        self.th_visible_bias = th.shared(self.visible_bias, 'th_visible_bias')
        self.th_hidden_group_biases = th.shared(self.hidden_group_biases, 'th_hidden_group_biases')
        self.th_weight_groups = th.shared(self.weight_groups, 'th_weight_groups')
        self.th_visible_layer_shape = th.shared(self.visible_layer_shape, 'th_visible_layer_shape')
        self.th_hidden_layer_shape = th.shared(self.hidden_layer_shape, 'th_hidden_layer_shape')

        self.__th_h_given_v = self.__init__th_h_given_v()
        self.__th_v_given_h = self.__init__th_v_given_h()
        self.__th_update_weights = self.__init__th_update_weights()
        self.__th_update_hidden_biases = self.__init__th_update_hidden_biases()
        self.__th_update_visible_bias = self.__init__th_update_visible_bias()

    def __init__th_h_given_v(self):
        th_h_given_v_input = T.tensor4("th_h_given_v_input", dtype=th.config.floatX)
        th_h_given_v_output_pre_sigmoid = T.nnet.conv2d(th_h_given_v_input, self.th_weight_groups, border_mode='valid',
                                                        filter_flip=False) + \
                                          self.th_hidden_group_biases.dimshuffle('x', 0, 'x', 'x')
        th_h_given_v_output_sigmoid = self.__sigmoid(th_h_given_v_output_pre_sigmoid)
        th_h_given_v_output_sampled = self.theano_rng.binomial(size=th_h_given_v_output_sigmoid.shape,
                                                               # discrete: binomial
                                                               n=1,
                                                               p=th_h_given_v_output_sigmoid)

        op = th.function(
            inputs=[th_h_given_v_input],
            outputs=[th_h_given_v_output_sigmoid, th_h_given_v_output_sampled]
            # givens=[self.th_weight_groups,self.th_hidden_group_biases]
        )

        return op

    def __init__th_v_given_h(self):
        th_v_given_h_input = T.tensor4('th_v_given_h_input', dtype=th.config.floatX)

        th_v_given_h_output_pre_sample = T.nnet.conv2d(th_v_given_h_input, self.th_weight_groups.swapaxes(0, 1),
                                                       border_mode='full') + self.th_visible_bias
        th_v_given_h_output_sampled = self.theano_rng.normal(avg=th_v_given_h_output_pre_sample,
                                                             size=self.visible_layer_shape)

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
        th_learning_rate = T.scalar('th_learning_rate', dtype=th.config.floatX)

        th_weight_update = th_learning_rate * (1./self.th_hidden_layer_shape.get_value()[2]**2) * \
                           (T.nnet.conv2d(th_input_sample, th_h0_pre_sample.swapaxes(0,1), filter_flip=False, border_mode='valid') -
                            T.nnet.conv2d(th_v1_sampled, th_h1_pre_sample.swapaxes(0,1), filter_flip=False, border_mode='valid'))

        op = th.function(
            inputs=[th_input_sample, th_h0_pre_sample, th_v1_sampled, th_h1_pre_sample, th_learning_rate],
            outputs=th_weight_update,
            updates=[(self.th_weight_groups, self.th_weight_groups + th_weight_update.swapaxes(0,1))]
        )

        return op

    def __init__th_update_hidden_biases(self):
        th_h0_pre_sample = T.tensor4('th_h0_pre_sample', dtype=th.config.floatX)
        th_h1_pre_sample = T.tensor4('th_h1_pre_sample', dtype=th.config.floatX)
        th_learning_rate = T.scalar('th_learning_rate', dtype=th.config.floatX)

        th_diff = th_h0_pre_sample - th_h1_pre_sample
        th_bias_updates = th_learning_rate * (1./self.th_hidden_layer_shape.get_value()[2]**2) * \
                          (th_diff.sum(axis=(2, 3)))

        op = th.function(
            inputs=[th_h0_pre_sample, th_h1_pre_sample, th_learning_rate],
            outputs=[th_bias_updates[0]],
            updates=[(self.th_hidden_group_biases, self.th_hidden_group_biases + th_bias_updates[0])]
        )

        return op

    def __init__th_update_visible_bias(self):
        th_v0 = T.tensor4('th_v0', dtype=th.config.floatX)
        th_v1 = T.tensor4('th_v1', dtype=th.config.floatX)
        th_learning_rate = T.scalar('th_learning_rate', dtype=th.config.floatX)

        th_diff = th_v0 - th_v1
        th_bias_update = th_learning_rate * (1./self.th_visible_layer_shape.get_value()[2]**2) * \
                         (th_diff.sum())

        op = th.function(
            inputs=[th_v0, th_v1, th_learning_rate],
            outputs=[th_bias_update],
            updates=[(self.th_visible_bias, self.th_visible_bias + th_bias_update)]
        )

        return op
