from abc import ABCMeta, abstractmethod
from layer import BinaryVisibleNonPooledLayer


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

    def train_layer_on_batch(self, train_input, layer_idx_to_train=-1):
        actual_train_input = train_input.copy()
        if layer_idx_to_train is -1:
            layer_idx_to_train += len(self.layers)

        if layer_idx_to_train is not 0:
            actual_train_input = self.infer_hid_given_vis(actual_train_input, layer_idx_to_train - 1)[1]

        return self.layers[layer_idx_to_train].train_on_minibatch(actual_train_input)

    def add_layer(self, new_layer_input_shape, new_layer_output_shape, learning_rate=0.01, target_sparsity=0.1,
                  sparsity_learning_rate=0.1):
        layer = self.create_next_layer(new_layer_input_shape, new_layer_output_shape, learning_rate, target_sparsity,
                                       sparsity_learning_rate)
        self.layers.append(layer)

    @abstractmethod
    def create_next_layer(self, new_layer_input_shape, new_layer_output_shape, learning_rate, target_sparsity,
                          sparsity_learning_rate):
        pass


class BinaryVisibleNonPooledDbn(AbstractDbn):
    def __init__(self, first_layer=None, saved_state=None):
        super(BinaryVisibleNonPooledDbn, self).__init__(first_layer, saved_state)

    def create_next_layer(self, new_layer_input_shape, new_layer_output_shape, learning_rate, target_sparsity,
                          sparsity_learning_rate):
        if len(self.layers) is 0:
            return BinaryVisibleNonPooledLayer(vis_unit_shape=new_layer_input_shape,
                                               hid_unit_shape=new_layer_output_shape,
                                               learning_rate=learning_rate,
                                               target_sparsity=target_sparsity,
                                               sparsity_learning_rate=sparsity_learning_rate)
        else:
            top_layer = self.layers[-1]
            next_layer_vis_units = top_layer.get_hidden_units()
            return BinaryVisibleNonPooledLayer(vis_unit_shape=next_layer_vis_units.get_shape(),
                                               hid_unit_shape=new_layer_output_shape,
                                               pre_set_vis_units=next_layer_vis_units,
                                               learning_rate=learning_rate, target_sparsity=target_sparsity,
                                               sparsity_learning_rate=sparsity_learning_rate)
