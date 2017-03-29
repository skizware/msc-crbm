class RbmGibbsSampler(object):
    def __init__(self, rbm_layer):
        self.rbm_layer =rbm_layer

    def sample_from_distribution(self, initial_hidden_val):
        vis_infer, vis_sampled = self.rbm_layer.infer_vis_given_hid(initial_hidden_val)
        hid_infer, hid_sampled = self.rbm_layer.infer_hid_given_vis(vis_sampled)

        return [vis_infer, vis_sampled, hid_infer, hid_sampled]