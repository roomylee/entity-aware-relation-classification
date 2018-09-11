import tensorflow as tf


def char_cnn(inputs, filter_sizes, num_filters, char_embedding_size, word_length, sequence_length):
    pooled_outputs = []
    for i, filter_size in enumerate(filter_sizes):
        with tf.name_scope("conv-maxpool-%s" % filter_size):
            # Convolution Layer
            filter_shape = [1, filter_size, char_embedding_size, num_filters]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
            conv = tf.nn.conv2d(inputs, W, strides=[1, 1, 1, 1], padding="VALID", name="conv")
            # Apply nonlinearity
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
            # Maxpooling over the outputs
            pooled = tf.nn.max_pool(h, ksize=[1, 1, word_length - filter_size + 1, 1],
                                    strides=[1, 1, 1, 1], padding='VALID', name="pool")
            pooled_outputs.append(pooled)
    # Combine all the pooled features
    num_filters_total = num_filters * len(filter_sizes)
    h_pool = tf.concat(pooled_outputs, 3)
    h_pool_flat = tf.reshape(h_pool, [-1, sequence_length, num_filters_total])

    return h_pool_flat


def highway_fc_layer(inputs, hidden_layer_size, activation=tf.nn.relu):
    '''
    The function used to crate Highway fully connected layer in the network.

    Inputs: input - data input
            hidden_layer_size - number of neurons in the hidden layers (highway layers)
            carry_b -  value for the carry bias used in transform gate
            activation - non-linear function used at this layer
    '''
    # Step 1. Define weights and biases for the activation gate
    weights_normal = tf.Variable(tf.truncated_normal([hidden_layer_size, hidden_layer_size], stddev=0.1))
    bias_normal = tf.Variable(tf.constant(0.1, shape=[hidden_layer_size]))

    # Step 2. Define weights and biases for the transform gate
    weights_transform = tf.Variable(tf.truncated_normal([hidden_layer_size, hidden_layer_size], stddev=0.1))
    bias_transform = tf.Variable(tf.constant(-2.0, shape=[hidden_layer_size]))

    # Step 3. calculate activation gate
    H = activation(tf.tensordot(inputs, weights_normal, axes=1) + bias_normal, name="Input_gate")
    # Step 4. calculate transform game
    T = tf.nn.sigmoid(tf.tensordot(inputs, weights_transform, axes=1) + bias_transform, name="T_gate")
    # Step 5. calculate carry get (1 - T)
    C = tf.subtract(1.0, T, name='C_gate')
    # y = (H * T) + (x * C)
    # Final step 6. campute the output from the highway fully connected layer
    y = tf.add(tf.multiply(H, T), tf.multiply(inputs, C), name='output_highway')
    return y