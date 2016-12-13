def sample_v_given_h(self, hidden_groups):
    	pre_sampled = np.zeros(self.visible_layer_shape)
    	for group in range(1, self.numBases):
    		#scipy.signal.convolve2d(matX, matY, mode)
    		pre_sampled += convolve2d(hidden_groups[group], self.weight_groups[group], mode='full')

        #shared visible bias
    	pre_sampled += self.visible_bias

        #rng is numpy.random.RandomState.normal(loc, scale, size)
        #It samples from normal distribution
        #loc is the mean
        #scale is the std deviation
        sampled = self.rng.normal(loc=pre_sampled, scale=1, size=self.visible_layer_shape)

    	#returning both pre-sampled and sampled values respectively for now
    	return [pre_sampled, sampled]