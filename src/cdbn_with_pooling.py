import crbm_with_pooling
import numpy as np


class Dbn(object):
    def __init__(self, crbm_layers):
        self.layers = crbm_layers

    def sample_h_given_v(self, input, starting_layer=0, ending_layer=-1):
        if ending_layer == -1:
            ending_layer = len(self.layers) - 1

        output_pre_sample = output_sample = running_input = input.copy()
        p = None
        p_sample = None
        for layer in self.layers[starting_layer:ending_layer]:
            output_pre_sample, output_sample, p, p_sample, _ = _, _, _, running_input, _ = layer.sample_h_given_v(running_input)

        return [output_pre_sample, output_sample, p, p_sample]

    def sample_v_given_h(self, hidden_input, starting_layer=-1, ending_layer=0, filter_vis=False):
        if starting_layer == -1:
            starting_layer = len(self.layers) - 1

        running_input = hidden_input
        numLayersRemaining = len(self.layers[ending_layer:starting_layer+1])
        for layer in reversed(self.layers[ending_layer:starting_layer + 1]):
            if numLayersRemaining < len(self.layers[ending_layer:starting_layer+1]):
                if filter_vis:
                    none_h = np.zeros((running_input.shape[0], running_input.shape[1], running_input.shape[2]*layer.pooling_ratio, running_input.shape[3]*layer.pooling_ratio))
                    running_input = layer.sample_h_given_p_none_h(running_input, none_h)[1]
                else:
                    running_input = layer.sample_h_given_p(running_input)[1]
            output_pre_sample, output_sample = _, running_input = layer.sample_v_given_h(running_input)
            numLayersRemaining -= 1

        return [output_pre_sample, output_sample]

    def recreation_at_layer(self, input_sample, layer=-1):
        if layer == -1:
            layer = len(self.target_dbn.layers)

        return self.sample_v_given_h(self.sample_h_given_v(input_sample, ending_layer=layer+1)[1], starting_layer=layer)

    def contrastive_divergence(self, input_sample, layer_number):
        if (layer_number != 0):
            input_sample = self.sample_h_given_v(input_sample, ending_layer=layer_number - 1)

        return self.layers[layer_number].contrastive_divergence(input_sample)

    def getStateObject(self):
        layer_count = 0
        state_object = {}
        for layer in self.layers:
            state_object[layer_count] = layer.getStateObject()
            layer_count += 1
        return state_object

    def loadStateObject(self, stateObject):
        layers = []
        for layerNum in stateObject.keys():
            if stateObject[layerNum]['type'] == crbm_with_pooling.PooledCrbm:
                layer = crbm_with_pooling.PooledCrbm()
            else:
                layer = crbm_with_pooling.PooledBinaryCrbm()
            layer.loadStateObject(stateObject[layerNum])
            layers.append(layer)

        self.layers = layers
