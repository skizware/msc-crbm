import numpy as np
from scipy.signal import convolve2d

class crbm(object):

    def __init__(self, numBases=3, visible_layer_shape=(18,18), hidden_layer_shape=(8,8)):
    	self.rng = np.random.RandomState(1234)		
    	self.visible_layer_shape = visible_layer_shape
    	self.hidden_layer_shape = hidden_layer_shape
    	self.numBases = numBases
    	self.visible_bias = 0
    	self.hidden_group_biases = np.zeros(numBases, dtype=float)

    	a = 1. / 10000
        self.weight_groups = np.array(self.rng.uniform(  # initialize W uniformly
                low=-a,
                high=a,
                size=(self.__get_weight_matrix_shape())))

        print self.weight_groups.shape

    def getStateObject(self):
    	return {'hidden_layer_shape':self.hidden_layer_shape, 'visible_layer_shape':self.visible_layer_shape, \
    			'numBases':self.numBases, 'visible_bias':self.visible_bias, 'hidden_group_biases':self.hidden_group_biases, \
    			'weight_groups':self.weight_groups}

    def loadStateObject(self, stateObject):
    	self.hidden_layer_shape = stateObject['hidden_layer_shape']
    	self.visible_layer_shape = stateObject['visible_layer_shape']
    	self.numBases = stateObject['numBases']
    	self.visible_bias = stateObject['visible_bias']
    	self.hidden_group_biases = stateObject['hidden_group_biases']
    	self.weight_groups = stateObject['weight_groups']


    def contrastive_divergence(self, trainingSample):
    	h0_pre_sample, h0_sample = self.sample_h_given_v(trainingSample)
    	v1_pre_sample, v1_sample,\
    	h1_pre_sample, h1_sample = self.gibbs_hvh(h0_sample)

    	for group in range(0,self.numBases):
    		self.weight_groups[group] += 0.001*(1./self.hidden_layer_shape[0]**2)*(convolve2d(trainingSample, np.rot90(h0_pre_sample[group],2), mode='valid') -\
    											convolve2d(v1_sample, np.rot90(h1_pre_sample[group],2), mode='valid'))
    		self.hidden_group_biases[group] += 0.001*(1./self.hidden_layer_shape[0]**2)*(np.nansum(h0_pre_sample[group] - h1_pre_sample[group]))

        print "nansum trainingsample = " + str(np.nansum(trainingSample))
        print "nansum v1_sample = " + str(np.nansum(v1_sample))
    	self.visible_bias += 0.001*(1./self.visible_layer_shape[0]**2)*(np.nansum(trainingSample - v1_sample))



    def sample_h_given_v(self, inputMat):
        hidden_groups = np.ndarray((self.numBases,) + self.hidden_layer_shape)		
        for group in range(0, self.numBases):
            hidden_groups[group] = convolve2d(inputMat, np.rot90(self.weight_groups[group], 2), mode='valid') + self.hidden_group_biases[group]

        return [self.__sigmoid(hidden_groups), self.rng.binomial(size=hidden_groups.shape,   # discrete: binomial
                                       n=1,
                                       p=self.__sigmoid(hidden_groups))]

    def sample_v_given_h(self, hidden_groups):
    	result = np.zeros(self.visible_layer_shape)
    	for group in range(0, self.numBases):
    		#scipy.signal.convolve2d(matX, matY, mode)
    		result += convolve2d(hidden_groups[group], self.weight_groups[group], mode='full')


    	result += self.visible_bias

    	#rng = numpy.random.RandomState.normal(loc, scale, size)
    	#Samples from normal distribution
    	#loc = mean
    	#scale = std deviation
    	#returning both pre-sampled and sampled values for now
    	return [result, self.rng.normal(loc=result, scale=1, size=self.visible_layer_shape)]

    def gibbs_vhv(self, inputMat):
    	h_pre_sample, h_sample = self.sample_h_given_v(inputMat)
    	v_pre_sample, v_sample = self.sample_v_given_h(h_sample)
    	return [v_pre_sample, v_sample, h_pre_sample, h_sample]

    def gibbs_hvh(self, hidden_groups):
    	v_pre_sample, v_sample = self.sample_v_given_h(hidden_groups)
    	h_pre_sample, h_sample = self.sample_h_given_v(v_sample)
    	return [v_pre_sample, v_sample, h_pre_sample, h_sample]

    def __get_weight_matrix_shape(self):
    	return (self.numBases,) + \
    	(self.visible_layer_shape[0] - self.hidden_layer_shape[0] + 1,\
    	 self.visible_layer_shape[1] - self.hidden_layer_shape[1] + 1)

    def __sigmoid(self, x):
        return 1. / (1 + np.exp(-x))