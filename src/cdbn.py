

class Dbn(object):

    def __init__(self, crbm_layers):
        self.layers = crbm_layers

    def sample_h_given_v(self, input, starting_layer=0, ending_layer=-1):
        if ending_layer == -1:
            ending_layer = len(self.layers) - 1

        output_sample = running_input = input.copy()
        for layer in self.layers[starting_layer:ending_layer+1]:
            output_pre_sample, output_sample = _, running_input = layer.sample_h_given_v(running_input)

        return [output_pre_sample, output_sample]

    def sample_v_given_h(self, hidden_input, starting_layer = -1, ending_layer=0):
        if starting_layer == -1:
            starting_layer = len(self.layers) - 1

        running_input = hidden_input
        for layer in reversed(self.layers[ending_layer:starting_layer+1]):
            output_pre_sample, output_sample = _, running_input = layer.sample_v_given_h(running_input)

        return [output_pre_sample, output_sample]

    def recreation_at_layer(self, input_sample, layer=-1):
        if layer == -1:
            layer = len(self.target_dbn.layers) - 1

        return self.sample_v_given_h(self.sample_h_given_v(input_sample, ending_layer=layer)[1], starting_layer=layer)

    def contrastive_divergence(self, input_sample, layer_number):
        if(layer_number != 0):
            input_sample = self.sample_h_given_v(input_sample, ending_layer=layer_number-1)

        return self.layers[layer_number].contrastive_divergence(input_sample)

