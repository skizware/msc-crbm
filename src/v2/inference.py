import theano as th
import theano.tensor as T
import numpy as np
from abc import ABCMeta, abstractmethod


class AbstractNonPooledInference:
    __metaclass__ = ABCMeta

    def __init__(self, rbm_layer):
        self.rbm_layer = rbm_layer
        self.th_infer_hid_given_vis = self.__init__th_infer_hid_given_vis()

    @abstractmethod
    def infer_vis_given_hid(self, hid_input):
        pass

    def infer_hid_given_vis(self, vis_input):
        return self.th_infer_hid_given_vis(vis_input)

    def __init__th_infer_hid_given_vis(self):
        th_h_given_v_input = T.tensor4("th_h_given_v_input", dtype=th.config.floatX)
        th_h_given_v_output_pre_sigmoid = T.nnet.conv2d(th_h_given_v_input, self.rbm_layer.get_weight_matrix(),
                                                        border_mode='valid',
                                                        filter_flip=False) + \
                                          self.rbm_layer.get_hidden_biases().dimshuffle('x', 0, 'x', 'x')
        th_h_given_v_output_sigmoid = sigmoid(th_h_given_v_output_pre_sigmoid)
        th_h_given_v_output_sampled = self.rbm_layer.get_rng().binomial(size=th_h_given_v_output_sigmoid.shape,
                                                                        n=1,
                                                                        p=th_h_given_v_output_sigmoid)

        op = th.function(
            inputs=[th_h_given_v_input],
            outputs=[th_h_given_v_output_sigmoid, th_h_given_v_output_sampled]
        )

        return op


class NonPooledInferenceGaussianVisible(AbstractNonPooledInference):
    def __init__(self, rbm_layer):
        super(NonPooledInferenceGaussianVisible, self).__init__(rbm_layer)
        self.th_infer_vis_given_hid = self.__init__th_infer_vis_given_hid()

    def infer_vis_given_hid(self, hid_input):
        return self.th_infer_vis_given_hid(hid_input)

    def __init__th_infer_vis_given_hid(self):
        th_v_given_h_input = T.tensor4('th_v_given_h_input', dtype=th.config.floatX)

        th_v_given_h_output_pre_sample = T.nnet.conv2d(th_v_given_h_input,
                                                       self.rbm_layer.get_weight_matrix().swapaxes(0, 1),
                                                       border_mode='full') + self.rbm_layer.get_visible_bias()
        th_v_given_h_output_sampled = self.rbm_layer.get_rng().normal(avg=th_v_given_h_output_pre_sample,
                                                                      size=th_v_given_h_output_pre_sample.shape)

        op = th.function(
            inputs=[th_v_given_h_input],
            outputs=[th_v_given_h_output_pre_sample, th_v_given_h_output_sampled]
        )

        return op


class NonPooledInferenceBinaryVisible(AbstractNonPooledInference):
    def __init__(self, rbm_layer):
        super(NonPooledInferenceBinaryVisible, self).__init__(rbm_layer)
        self.th_infer_vis_given_hid = self.__init__th_infer_vis_given_hid()

    def infer_vis_given_hid(self, hid_input):
        return self.th_infer_vis_given_hid(hid_input)

    def __init__th_infer_vis_given_hid(self):
        th_v_given_h_input = T.tensor4('th_v_given_h_input', dtype=th.config.floatX)

        th_v_given_h_output_pre_sigmoid = T.nnet.conv2d(th_v_given_h_input,
                                                       self.rbm_layer.get_weight_matrix().swapaxes(0, 1),
                                                       border_mode='full') + self.rbm_layer.get_visible_bias()

        th_v_given_h_output_pre_sampled = sigmoid(th_v_given_h_output_pre_sigmoid)

        th_v_given_h_output_sampled = self.rbm_layer.get_rng().binomial(size=th_v_given_h_output_pre_sampled.shape,
                                                                        n=1,
                                                                        p=th_v_given_h_output_pre_sampled)

        op = th.function(
            inputs=[th_v_given_h_input],
            outputs=[th_v_given_h_output_pre_sampled, th_v_given_h_output_sampled]
        )

        return op


def sigmoid(x):
    return 1. / (1 + np.exp(-x))
