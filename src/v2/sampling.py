class RbmGibbsSampler(object):
    def __init__(self, rbm_layer):
        self.rbm_layer = rbm_layer

    def sample_from_distribution(self, initial_hidden_val):
        vis_infer, vis_sampled = self.rbm_layer.infer_vis_given_hid(initial_hidden_val)
        hid_infer, hid_sampled = self.rbm_layer.infer_hid_given_vis(vis_sampled)[0:2]

        return [vis_infer, vis_sampled, hid_infer, hid_sampled]


class RbmPersistentGibbsSampler(RbmGibbsSampler):
    def __init__(self, rbm_layer):
        super(RbmPersistentGibbsSampler, self).__init__(rbm_layer)
        self.value = None

    def sample_from_distribution(self, initial_hidden_val):
        if self.value is None:
            self.value = initial_hidden_val

        vis_infer, vis_sampled, hid_infer, hid_sampled = super(RbmPersistentGibbsSampler,
                                                               self).sample_from_distribution(self.value)

        self.value = hid_sampled

        return [vis_infer, vis_sampled, hid_infer, hid_sampled]

    def clear_value(self):
        self.value = None
