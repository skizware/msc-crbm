from abc import ABCMeta, abstractmethod
import numpy as np
from layer import BinaryVisibleNonPooledLayer, GaussianVisibleNonPooledLayer, BinaryVisiblePooledLayer, AbstractPooledLayer,\
    GaussianVisiblePooledLayer, BinaryVisibleNonPooledPersistentSamplerChainLayer, BinaryVisiblePooledPersistentSamplerChainLayer,\
    GaussianVisibleNonPooledPersistentSamplerChainLayer, GaussianVisiblePooledPersistentSamplerChainLayer
from layer import KEY_VIS_SHAPE, KEY_HID_SHAPE, KEY_LAYER_TYPE, KEY_LEARNING_RATE, KEY_HID_BIASES, \
    KEY_SPARSITY_LEARNING_RATE, KEY_TARGET_SPARSITY, KEY_VIS_BIAS, KEY_WEIGHT_MATRIX, KEY_POOLING_RATIO

KEY_DBN_TYPE = 'type'

KEY_LAYERS = 'layers'


class AbstractDbn(object):
    __metaclass__ = ABCMeta

    def __init__(self, first_layer=None, saved_state=None):
        self.layers = []
        if first_layer is not None:
            self.layers.append(first_layer)

    def infer_hid_given_vis(self, vis_input, end_layer_index_incl=-1):
        if end_layer_index_incl is -1:
            end_layer_index_incl += len(self.layers)

        ret = self.layers[0].infer_hid_given_vis(vis_input)
        for layer_index in range(1, end_layer_index_incl + 1):
            ret = self.layers[layer_index].infer_hid_given_vis()

        return ret

    def infer_vis_given_hid(self, hid_input, start_layer_index=-1, end_layer_index_incl=-1):
        if end_layer_index_incl is -1:
            end_layer_index_incl = 0
        if start_layer_index is -1:
            start_layer_index += len(self.layers)

        ret = self.layers[start_layer_index].infer_vis_given_hid(hid_input)
        for layer_idx in xrange(start_layer_index - 1, end_layer_index_incl - 1, -1):
            ret = self.layers[layer_idx].infer_vis_given_hid()

        return ret

    def get_learned_state(self):
        learned_state = {KEY_LAYERS: {},
                         KEY_DBN_TYPE: type(self)}
        for layer_idx in xrange(0, len(self.layers)):
            learned_state[KEY_LAYERS][layer_idx] = self.layers[layer_idx].get_state_object()

        return learned_state

    def train_layer_on_batch(self, train_input, layer_idx_to_train=-1):
        actual_train_input = train_input.copy()
        if layer_idx_to_train is -1:
            layer_idx_to_train += len(self.layers)

        if layer_idx_to_train is not 0:
            actual_train_input = self.infer_hid_given_vis(actual_train_input, layer_idx_to_train - 1)[-1]

        return self.layers[layer_idx_to_train].train_on_minibatch(actual_train_input)

    def add_layer(self, new_layer_input_shape, new_layer_output_shape, learning_rate=0.01, target_sparsity=0.1,
                  sparsity_learning_rate=0.1, pooling_ratio=0):
        layer = self.create_next_layer(new_layer_input_shape, new_layer_output_shape, learning_rate, target_sparsity,
                                       sparsity_learning_rate, pooling_ratio)
        self.layers.append(layer)

    def get_features(self, dbn_input, layers_to_use=None):
        features = None
        if layers_to_use is None:
            layers_to_use = range(0, len(self.layers))

        top_layer_to_use = max(layers_to_use)
        layer_in = dbn_input
        for layer_idx in xrange(0, top_layer_to_use+1):
            inferred_val_idx = 2 if isinstance(self.layers[layer_idx], AbstractPooledLayer) else 0
            layer_in = self.layers[layer_idx].infer_hid_given_vis(layer_in)
            if layer_idx in layers_to_use:
                layer_shape = layer_in[inferred_val_idx].shape
                if features is None:
                    features = layer_in[inferred_val_idx].reshape(layer_shape[0], layer_shape[1] * layer_shape[2] * layer_shape[3])
                else:
                    np.concatenate((features, layer_in[inferred_val_idx].reshape(layer_shape[0], layer_shape[1] * layer_shape[2] * layer_shape[3])), axis=1)
            layer_in = layer_in[inferred_val_idx + 1]

        return np.asarray(features)

    @abstractmethod
    def create_next_layer(self, new_layer_input_shape, new_layer_output_shape, learning_rate, target_sparsity,
                          sparsity_learning_rate, pooling_ratio):
        pass


class BinaryVisibleDbn(AbstractDbn):
    def __init__(self, first_layer=None, saved_state=None):
        super(BinaryVisibleDbn, self).__init__(first_layer, saved_state)

    def create_next_layer(self, new_layer_input_shape, new_layer_output_shape, learning_rate, target_sparsity,
                          sparsity_learning_rate, pooling_ratio):
        if len(self.layers) is 0:
            if pooling_ratio is not 0:
                return BinaryVisiblePooledLayer(vis_unit_shape=new_layer_input_shape,
                                               hid_unit_shape=new_layer_output_shape,
                                               learning_rate=learning_rate,
                                               target_sparsity=target_sparsity,
                                               sparsity_learning_rate=sparsity_learning_rate,
                                               pooling_ratio=pooling_ratio)
            else:
                return BinaryVisibleNonPooledLayer(vis_unit_shape=new_layer_input_shape,
                                                   hid_unit_shape=new_layer_output_shape,
                                                   learning_rate=learning_rate,
                                                   target_sparsity=target_sparsity,
                                                   sparsity_learning_rate=sparsity_learning_rate)
        else:
            top_layer = self.layers[-1]
            if pooling_ratio is not None:
                next_layer_vis_units = top_layer.get_pool_units()
                return BinaryVisiblePooledLayer(vis_unit_shape=next_layer_vis_units.get_shape(),
                                                hid_unit_shape=new_layer_output_shape,
                                                pre_set_vis_units=next_layer_vis_units,
                                                learning_rate=learning_rate, target_sparsity=target_sparsity,
                                                sparsity_learning_rate=sparsity_learning_rate,
                                                pooling_ratio=pooling_ratio)
            else:
                next_layer_vis_units = top_layer.get_hidden_units()
                return BinaryVisibleNonPooledLayer(vis_unit_shape=next_layer_vis_units.get_shape(),
                                                   hid_unit_shape=new_layer_output_shape,
                                                   pre_set_vis_units=next_layer_vis_units,
                                                   learning_rate=learning_rate, target_sparsity=target_sparsity,
                                                   sparsity_learning_rate=sparsity_learning_rate)


class BinaryVisibleDbnPersistentChainSampling(AbstractDbn):
    def __init__(self, first_layer=None, saved_state=None):
        super(BinaryVisibleDbnPersistentChainSampling, self).__init__(first_layer, saved_state)

    def create_next_layer(self, new_layer_input_shape, new_layer_output_shape, learning_rate, target_sparsity,
                          sparsity_learning_rate, pooling_ratio):
        if len(self.layers) is 0:
            if pooling_ratio is not 0:
                return BinaryVisiblePooledPersistentSamplerChainLayer(vis_unit_shape=new_layer_input_shape,
                                               hid_unit_shape=new_layer_output_shape,
                                               learning_rate=learning_rate,
                                               target_sparsity=target_sparsity,
                                               sparsity_learning_rate=sparsity_learning_rate,
                                               pooling_ratio=pooling_ratio)
            else:
                return BinaryVisibleNonPooledPersistentSamplerChainLayer(vis_unit_shape=new_layer_input_shape,
                                                   hid_unit_shape=new_layer_output_shape,
                                                   learning_rate=learning_rate,
                                                   target_sparsity=target_sparsity,
                                                   sparsity_learning_rate=sparsity_learning_rate)
        else:
            top_layer = self.layers[-1]
            if pooling_ratio is not 0:
                next_layer_vis_units = top_layer.get_pool_units()
                return BinaryVisiblePooledPersistentSamplerChainLayer(vis_unit_shape=next_layer_vis_units.get_shape(),
                                                hid_unit_shape=new_layer_output_shape,
                                                pre_set_vis_units=next_layer_vis_units,
                                                learning_rate=learning_rate, target_sparsity=target_sparsity,
                                                sparsity_learning_rate=sparsity_learning_rate,
                                                pooling_ratio=pooling_ratio)
            else:
                next_layer_vis_units = top_layer.get_hidden_units()
                return BinaryVisibleNonPooledPersistentSamplerChainLayer(vis_unit_shape=next_layer_vis_units.get_shape(),
                                                   hid_unit_shape=new_layer_output_shape,
                                                   pre_set_vis_units=next_layer_vis_units,
                                                   learning_rate=learning_rate, target_sparsity=target_sparsity,
                                                   sparsity_learning_rate=sparsity_learning_rate)


class GaussianVisibleDbn(AbstractDbn):
    def __init__(self, first_layer=None, saved_state=None):
        super(GaussianVisibleDbn, self).__init__(first_layer, saved_state)

    def create_next_layer(self, new_layer_input_shape, new_layer_output_shape, learning_rate, target_sparsity,
                          sparsity_learning_rate, pooling_ratio):
        if len(self.layers) is 0:
            if pooling_ratio is not None and pooling_ratio is not 0:
                return GaussianVisiblePooledLayer(vis_unit_shape=new_layer_input_shape,
                                               hid_unit_shape=new_layer_output_shape,
                                               learning_rate=learning_rate,
                                               target_sparsity=target_sparsity,
                                               sparsity_learning_rate=sparsity_learning_rate,
                                               pooling_ratio=pooling_ratio)
            else:
                return GaussianVisibleNonPooledLayer(vis_unit_shape=new_layer_input_shape,
                                                   hid_unit_shape=new_layer_output_shape,
                                                   learning_rate=learning_rate,
                                                   target_sparsity=target_sparsity,
                                                   sparsity_learning_rate=sparsity_learning_rate)
        else:
            top_layer = self.layers[-1]
            if pooling_ratio is not None:
                next_layer_vis_units = top_layer.get_pool_units()
                return BinaryVisiblePooledLayer(vis_unit_shape=next_layer_vis_units.get_shape(),
                                                hid_unit_shape=new_layer_output_shape,
                                                pre_set_vis_units=next_layer_vis_units,
                                                learning_rate=learning_rate, target_sparsity=target_sparsity,
                                                sparsity_learning_rate=sparsity_learning_rate,
                                                pooling_ratio=pooling_ratio)
            else:
                next_layer_vis_units = top_layer.get_hidden_units()
                return BinaryVisibleNonPooledLayer(vis_unit_shape=next_layer_vis_units.get_shape(),
                                                   hid_unit_shape=new_layer_output_shape,
                                                   pre_set_vis_units=next_layer_vis_units,
                                                   learning_rate=learning_rate, target_sparsity=target_sparsity,
                                                   sparsity_learning_rate=sparsity_learning_rate)


class GaussianVisibleDbnPersistentChainSampling(AbstractDbn):
    def __init__(self, first_layer=None, saved_state=None):
        super(GaussianVisibleDbnPersistentChainSampling, self).__init__(first_layer, saved_state)

    def create_next_layer(self, new_layer_input_shape, new_layer_output_shape, learning_rate, target_sparsity,
                          sparsity_learning_rate, pooling_ratio):
        if len(self.layers) is 0:
            if pooling_ratio is not 0:
                return GaussianVisiblePooledPersistentSamplerChainLayer(vis_unit_shape=new_layer_input_shape,
                                               hid_unit_shape=new_layer_output_shape,
                                               learning_rate=learning_rate,
                                               target_sparsity=target_sparsity,
                                               sparsity_learning_rate=sparsity_learning_rate,
                                               pooling_ratio=pooling_ratio)
            else:
                return GaussianVisibleNonPooledPersistentSamplerChainLayer(vis_unit_shape=new_layer_input_shape,
                                                   hid_unit_shape=new_layer_output_shape,
                                                   learning_rate=learning_rate,
                                                   target_sparsity=target_sparsity,
                                                   sparsity_learning_rate=sparsity_learning_rate)
        else:
            top_layer = self.layers[-1]
            if pooling_ratio is not 0:
                next_layer_vis_units = top_layer.get_pool_units()
                return BinaryVisiblePooledPersistentSamplerChainLayer(vis_unit_shape=next_layer_vis_units.get_shape(),
                                                hid_unit_shape=new_layer_output_shape,
                                                pre_set_vis_units=next_layer_vis_units,
                                                learning_rate=learning_rate, target_sparsity=target_sparsity,
                                                sparsity_learning_rate=sparsity_learning_rate,
                                                pooling_ratio=pooling_ratio)
            else:
                next_layer_vis_units = top_layer.get_hidden_units()
                return BinaryVisibleNonPooledPersistentSamplerChainLayer(vis_unit_shape=next_layer_vis_units.get_shape(),
                                                   hid_unit_shape=new_layer_output_shape,
                                                   pre_set_vis_units=next_layer_vis_units,
                                                   learning_rate=learning_rate, target_sparsity=target_sparsity,
                                                   sparsity_learning_rate=sparsity_learning_rate)


class BinaryVisibleNonPooledDbn(object):
    print "hellow"

class DbnFromStateBuilder(object):
    @staticmethod
    def init_dbn(learned_state):
        dbn = learned_state[KEY_DBN_TYPE]()
        for layer_idx in learned_state[KEY_LAYERS]:
            layer_state = learned_state[KEY_LAYERS][layer_idx]
            pooling_ratio = layer_state[KEY_POOLING_RATIO] if layer_state.has_key(KEY_POOLING_RATIO) else None
            dbn.add_layer(layer_state[KEY_VIS_SHAPE], layer_state[KEY_HID_SHAPE], layer_state[KEY_LEARNING_RATE],
                          layer_state[KEY_TARGET_SPARSITY], layer_state[KEY_SPARSITY_LEARNING_RATE], pooling_ratio)
            dbn.layers[layer_idx].set_internal_state(layer_state)

        return dbn
