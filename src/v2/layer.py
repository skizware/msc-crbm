from abc import ABCMeta, abstractmethod
from unit import LayerUnits
from inference import NonPooledInferenceBinaryVisible, NonPooledInferenceGaussianVisible
import numpy as np

import theano as th
from theano.tensor.shared_randomstreams import RandomStreams


class AbstractLayer:
    __metaclass__ = ABCMeta

    """
    Abstract base class for a layer in a DBN.

    Layers have visible units, and hidden units.
    Visible Units have shape (batch size, num channels, in-rows, in-cols)
    Hidden units have shape (batch size, num learned bases, out-rows, out-cols)
    """

    def __init__(self, vis_unit_shape, hid_unit_shape, pre_set_vis_units=None, pre_set_hid_units=None):
        self.__init_units(hid_unit_shape, vis_unit_shape, pre_set_hid_units, pre_set_vis_units)
        self.__batch_size = vis_unit_shape[0]
        self.__num_bases = hid_unit_shape[1]
        self.__vis_bias = 0.
        self.__hid_biases = np.zeros((self.__num_bases,), dtype=float)
        self.__rng = np.random.RandomState()

        init_weight_variance = 1. / 100
        self.__weight_matrix = np.array(self.__rng.uniform(  # initialize W uniformly
            low=-init_weight_variance,
            high=init_weight_variance,
            size=(self.__get_weight_matrix_shape(vis_unit_shape, hid_unit_shape))))

        self.__th_setup()
        self.init_inference_proc()

    def init_inference_proc(self):
        raise NotImplementedError("Do not use the abstract base class")

    def infer_hid_given_vis(self, vis_input=None):
        if vis_input is None:
            vis_input = self.__vis_units.get_value()
            assert vis_input is not None, "Error - no value set for visible units"

        ret = self.inference_proc.infer_hid_given_vis(vis_input)
        self.__hid_units.set_value(ret[1])
        return ret

    def infer_vis_given_hid(self, hid_input=None):
        if hid_input is None:
            hid_input = self.__hid_units.get_value()
            assert hid_input is not None, "Error - no value set for hidden units"

        ret = self.inference_proc.infer_vis_given_hid(hid_input)
        self.__vis_units.set_value(ret[1])
        return ret

    def __init_units(self, hid_unit_shape, vis_unit_shape, pre_set_hid_units, pre_set_vis_units):
        if pre_set_vis_units is not None:
            assert (
                vis_unit_shape == pre_set_vis_units.get_shape(), 'Unit shape mismatch with pre defined visible units')
            self.__vis_units = pre_set_vis_units
        else:
            self.__vis_units = LayerUnits(vis_unit_shape)
        if pre_set_hid_units is not None:
            assert (
                hid_unit_shape == pre_set_hid_units.get_shape(), 'Unit shape mismatch with pre defined hidden units')
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

    # ==============================Theano Specifics================================== #
    def __th_setup(self):
        self.__theano_rng = RandomStreams(self.__rng.randint(2 ** 30))
        self.__th_vis_bias = th.shared(self.__vis_bias)
        self.__th_hid_biases = th.shared(self.__hid_biases)
        self.__th_weight_matrix = th.shared(self.__weight_matrix)
        self.__th_vis_shape = th.shared(self.__vis_units.get_shape())
        self.__th_hid_shape = th.shared(self.__hid_units.get_shape())

    def get_weight_matrix(self):
        return self.__th_weight_matrix

    def get_hidden_biases(self):
        return self.__th_hid_biases

    def get_visible_bias(self):
        return self.__th_vis_bias

    def get_rng(self):
        return self.__theano_rng


class BinaryVisibleNonPooledLayer(AbstractLayer):
    def __init__(self, vis_unit_shape, hid_unit_shape, pre_set_vis_units=None, pre_set_hid_units=None):
        super(BinaryVisibleNonPooledLayer, self).__init__(vis_unit_shape, hid_unit_shape, pre_set_vis_units,
                                                          pre_set_hid_units)

    def init_inference_proc(self):
        self.inference_proc = NonPooledInferenceBinaryVisible(self)


class GaussianVisibleNonPooledLayer(AbstractLayer):
    def __init__(self, vis_unit_shape, hid_unit_shape, pre_set_vis_units=None, pre_set_hid_units=None):
        super(GaussianVisibleNonPooledLayer, self).__init__(vis_unit_shape, hid_unit_shape, pre_set_vis_units,
                                                            pre_set_hid_units)

    def init_inference_proc(self):
        self.inference_proc = NonPooledInferenceGaussianVisible(self)
