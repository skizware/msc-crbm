import numpy as np
import theano as th
from theano import tensor as T
import datetime
from theano.tensor.shared_randomstreams import RandomStreams

import tensorflow as tf
import tensorflow.contrib.distributions as distrib

NUM_PER_BATCH = 4

init_weight_variance = 1. / 100
rng = np.random.RandomState()
weight_matrix = np.array(rng.uniform(  # initialize W uniformly
    low=-init_weight_variance,
    high=init_weight_variance,
    size=(300, 1, 6, 6)))
hid_biases = np.zeros((300,), dtype=float)

tf_weight_matrix = tf.Variable(weight_matrix, dtype=tf.float32)
tf_hid_bias = tf.Variable(hid_biases, dtype=tf.float32)
tf_vis_bias = tf.Variable(0., dtype=tf.float32)

th_weight_matrix = th.shared(weight_matrix)
th_hid_biases = th.shared(hid_biases)
th_vis_bias = th.shared(0.)

theano_rng = RandomStreams(rng.randint(2 ** 30))


def sigmoid(x):
    return 1. / (1 + np.exp(-x))


"""def tf_prop_up():
    tf_input = tf.placeholder(tf.float32, name='tf_input')
    tf_h_given_v_output_pre_sigmoid = tf.nn.conv2d(input=tf_input, filter=tf_weight_matrix, strides=[1, 1, 1, 1],
                                                   padding="VALID", use_cudnn_on_gpu=True,
                                                   data_format="NCHW")
    tf_h_given_v_output_pre_sigmoid = tf.nn.bias_add(tf_h_given_v_output_pre_sigmoid, tf_hid_bias)
    tf_h_given_v_output_sigmoid = tf.sigmoid(tf_h_given_v_output_pre_sigmoid)
    tf_h_given_v_output_sampled = distrib.Binomial(total_count=1, probs=tf_h_given_v_output_sigmoid).sample(sample_shape=tf_h_given_v_output_sigmoid.shape)

    return tf_h_given_v_output_sigmoid, tf_h_given_v_output_sampled
"""

def th_prop_up(th_input):
    # th_input = T.tensor4('th_input', dtype=th.config.floatX)
    th_h_given_v_output_pre_sigmoid = T.nnet.conv2d(th_input, th_weight_matrix,
                                                    border_mode='valid',
                                                    filter_flip=False) + \
                                      th_hid_biases.dimshuffle('x', 0, 'x', 'x')
    th_h_given_v_output_sigmoid = sigmoid(th_h_given_v_output_pre_sigmoid)
    th_h_given_v_output_sampled = theano_rng.binomial(size=th_h_given_v_output_sigmoid.shape,
                                                      n=1,
                                                      p=th_h_given_v_output_sigmoid)

    return th_h_given_v_output_sigmoid, th_h_given_v_output_sampled


def th_prop_down(th_input):
    th_v_given_h_output_pre_sample = T.nnet.conv2d(th_input,
                                                   th_weight_matrix.swapaxes(0, 1),
                                                   border_mode='full') + th_vis_bias
    th_v_given_h_output_sampled = theano_rng.normal(avg=th_v_given_h_output_pre_sample,
                                                    size=th_v_given_h_output_pre_sample.shape)

    return th_v_given_h_output_pre_sample, th_v_given_h_output_sampled


def th_learn_op():
    th_input_batch = T.tensor4('th_input_batch', dtype=th.config.floatX)
    th_pos_hid_inf, th_pos_hid_sampled = th_prop_up(th_input_batch)
    # th_neg_vis_inf, th_neg_vis_sampled = th_prop_down(th_pos_hid_sampled)
    # th_neg_hid_inf, th_neg_hid_sampled = th_prop_up(th_neg_vis_sampled)

    # th_weight_updates = th_update_weight(th_input_batch.dimshuffle(1, 0, 2, 3), th_pos_hid_inf.dimshuffle(1, 0, 2, 3), th_neg_vis_sampled.dimshuffle(1, 0, 2, 3), th_neg_hid_inf.dimshuffle(1, 0, 2, 3))

    op = th.function(
        inputs=[th_input_batch],
        outputs=th_pos_hid_sampled
    )

    return op


def th_learn_op2():
    th_input_batch = T.tensor4('th_input_batch', dtype=th.config.floatX)
    # th_pos_hid_inf, th_pos_hid_sampled = th_prop_up(th_input_batch)
    th_neg_vis_inf, th_neg_vis_sampled = th_prop_down(th_input_batch)
    th_neg_hid_inf, th_neg_hid_sampled = th_prop_up(th_neg_vis_sampled)

    th_weight_updates = th_update_weight(th_neg_vis_inf.dimshuffle(1, 0, 2, 3), th_neg_hid_inf.dimshuffle(1, 0, 2, 3),
                                         th_neg_vis_sampled.dimshuffle(1, 0, 2, 3),
                                         th_neg_hid_inf.dimshuffle(1, 0, 2, 3))

    op = th.function(
        inputs=[th_input_batch],
        outputs=th_weight_updates
    )

    return op


def init__th_update_weights():
    """NB - Assumes the input for visible samples is (NumVisChannels, 1, R, C) and input for hidden samples is 
    (NumHidChannels, 1, R, C) """
    output, th_h0_pre_sample, th_h1_pre_sample, th_input_sample, th_v1_sampled = th_update_weight()

    op = th.function(
        inputs=[th_input_sample, th_h0_pre_sample, th_v1_sampled, th_h1_pre_sample],
        outputs=output
    )

    return op


def th_update_weight(th_input_sample, th_h0_pre_sample, th_v1_sampled, th_h1_pre_sample):
    output = (T.nnet.conv2d(th_input_sample, th_h0_pre_sample, filter_flip=False, border_mode='valid') -
              T.nnet.conv2d(th_v1_sampled, th_h1_pre_sample, filter_flip=False, border_mode='valid'))
    return output


op = th_learn_op()
op2 = th_learn_op2()

batchVis = np.ndarray(shape=(NUM_PER_BATCH, 1, 257, 115))
batchHid = np.ndarray(shape=(NUM_PER_BATCH, 300, 252, 110))

startTime = datetime.datetime.now()
res = op(batchVis)
res2 = op2(res)
endTime = datetime.datetime.now()
print res.shape
print res2.shape
print("Elapsed time = {}".format(endTime - startTime))

startTime = datetime.datetime.now()
for i in xrange(0, NUM_PER_BATCH):
    vis = batchVis[i:i + 1, :, :, :]
    hid = batchHid[i:i + 1, :, :, :]
    res = op(vis)
    res2 = op2(res)
    print res.shape
endTime = datetime.datetime.now()
print("Elapsed time = {}".format(endTime - startTime))


tf_input = tf.placeholder(tf.float32, name='tf_input')
tf_h_given_v_output_pre_sigmoid = tf.nn.conv2d(input=tf_input, filter=tf_weight_matrix, strides=[1, 1, 1, 1],
                                               padding="VALID", use_cudnn_on_gpu=True,
                                               data_format="NCHW")
tf_h_given_v_output_pre_sigmoid = tf.nn.bias_add(tf_h_given_v_output_pre_sigmoid, tf_hid_bias)
tf_h_given_v_output_sigmoid = tf.sigmoid(tf_h_given_v_output_pre_sigmoid)
tf_h_given_v_output_sampled = distrib.Binomial(total_count=1, probs=tf_h_given_v_output_sigmoid).sample(sample_shape=tf_h_given_v_output_sigmoid.shape)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
sess.run(tf_h_given_v_output_sampled, {tf_input: batchVis})
