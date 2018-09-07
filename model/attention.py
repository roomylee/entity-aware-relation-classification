import tensorflow as tf


def attention(inputs, attention_size, time_major=False, return_alphas=False):
    """
    Attention mechanism layer which reduces RNN/Bi-RNN outputs with Attention vector.

    The idea was proposed in the article by Z. Yang et al., "Hierarchical Attention Networks
     for Document Classification", 2016: http://www.aclweb.org/anthology/N16-1174.
    Variables notation is also inherited from the article

    Args:
        inputs: The Attention inputs.
            Matches outputs of RNN/Bi-RNN layer (not final state):
                In case of RNN, this must be RNN outputs `Tensor`:
                    If time_major == False (default), this must be a tensor of shape:
                        `[batch_size, max_time, cell.output_size]`.
                    If time_major == True, this must be a tensor of shape:
                        `[max_time, batch_size, cell.output_size]`.
                In case of Bidirectional RNN, this must be a tuple (outputs_fw, outputs_bw) containing the forward and
                the backward RNN outputs `Tensor`.
                    If time_major == False (default),
                        outputs_fw is a `Tensor` shaped:
                        `[batch_size, max_time, cell_fw.output_size]`
                        and outputs_bw is a `Tensor` shaped:
                        `[batch_size, max_time, cell_bw.output_size]`.
                    If time_major == True,
                        outputs_fw is a `Tensor` shaped:
                        `[max_time, batch_size, cell_fw.output_size]`
                        and outputs_bw is a `Tensor` shaped:
                        `[max_time, batch_size, cell_bw.output_size]`.
        attention_size: Linear size of the Attention weights.
        time_major: The shape format of the `inputs` Tensors.
            If true, these `Tensors` must be shaped `[max_time, batch_size, depth]`.
            If false, these `Tensors` must be shaped `[batch_size, max_time, depth]`.
            Using `time_major = True` is a bit more efficient because it avoids
            transposes at the beginning and end of the RNN calculation.  However,
            most TensorFlow data is batch-major, so by default this function
            accepts input and emits output in batch-major form.
        return_alphas: Whether to return attention coefficients variable along with layer's output.
            Used for visualization purpose.
    Returns:
        The Attention output `Tensor`.
        In case of RNN, this will be a `Tensor` shaped:
            `[batch_size, cell.output_size]`.
        In case of Bidirectional RNN, this will be a `Tensor` shaped:
            `[batch_size, cell_fw.output_size + cell_bw.output_size]`.
    """

    if isinstance(inputs, tuple):
        # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
        inputs = tf.concat(inputs, 2)

    if time_major:
        # (T,B,D) => (B,T,D)
        inputs = tf.array_ops.transpose(inputs, [1, 0, 2])

    hidden_size = inputs.shape[2].value  # D value - hidden size of the RNN layer

    # Trainable parameters
    w_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
    b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
    u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))

    with tf.name_scope('v'):
        # Applying fully connected layer with non-linear activation to each of the B*T timestamps;
        #  the shape of `v` is (B,T,D)*(D,A)=(B,T,A), where A=attention_size
        v = tf.tanh(tf.tensordot(inputs, w_omega, axes=1) + b_omega)

    # For each of the timestamps its vector of size A from `v` is reduced with `u` vector
    vu = tf.tensordot(v, u_omega, axes=1, name='vu')  # (B,T) shape
    alphas = tf.nn.softmax(vu, name='alphas')  # (B,T) shape

    # Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape
    output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)

    if not return_alphas:
        return output
    else:
        return output, alphas


def dot_self_attention(x, scale_factor):
    QK_T = tf.matmul(x, tf.transpose(x, [0, 2, 1]))
    att = tf.nn.softmax(QK_T / tf.sqrt(scale_factor))
    output = tf.matmul(att, x)
    return output


def add_self_attention(x):
    x1 = tf.layers.dense(x, 102)
    x2 = tf.layers.dense(x, 102)
    e = tf.layers.dense(tf.tanh(x1+x2), 102)
    att = tf.nn.softmax(e)
    output = tf.matmul(att, x)
    return output


def input_attention(x, e1, e2):
    A1 = tf.tanh(tf.matmul(x, tf.expand_dims(e1, -1)))
    A2 = tf.tanh(tf.matmul(x, tf.expand_dims(e2, -1)))
    A1 = tf.reshape(A1, [-1, 102])
    A2 = tf.reshape(A2, [-1, 102])
    alpha1 = tf.nn.softmax(A1)
    alpha2 = tf.nn.softmax(A2)
    alpha = (alpha1 + alpha2) / 2

    return tf.multiply(x, tf.expand_dims(alpha, -1))


def multihead_attention(query, key, value, dim_model=100, num_head=3):
    # Linear projections
    Q = tf.layers.dense(query, dim_model*num_head, activation=tf.nn.relu)  # (N, T_q, C)
    K = tf.layers.dense(key, dim_model*num_head, activation=tf.nn.relu)  # (N, T_k, C)
    V = tf.layers.dense(value, dim_model*num_head, activation=tf.nn.relu)  # (N, T_k, C)

    # Split and concat
    Q_ = tf.concat(tf.split(Q, num_head, axis=2), axis=0)  # (h*N, T_q, C/h)
    K_ = tf.concat(tf.split(K, num_head, axis=2), axis=0)  # (h*N, T_k, C/h)
    V_ = tf.concat(tf.split(V, num_head, axis=2), axis=0)  # (h*N, T_k, C/h)

    # Scaled Dot Product Attention
    with tf.variable_scope("scaled-dot-product-attention"):
        QK_T = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))
        QK_T /= K_.get_shape().as_list()[-1] ** 0.5
        att = tf.nn.softmax(QK_T)
        # outputs = tf.nn.dropout(att, 0.8)
        # Weighted sum
        outputs = tf.matmul(att, V_)  # ( h*N, T_q, C/h)
        # Restore shape
        outputs = tf.concat(tf.split(outputs, num_head, axis=0), axis=2)  # (N, T_q, C)

    # # Residual connection
    # outputs += query

    outputs = tf.layers.dense(outputs, dim_model * num_head, activation=tf.nn.relu)

    # Normalize
    outputs = normalize(outputs)  # (N, T_q, C)

    return outputs


def normalize(inputs,
              epsilon=1e-8,
              scope="ln",
              reuse=None):
    '''Applies layer normalization.

    Args:
      inputs: A tensor with 2 or more dimensions, where the first dimension has
        `batch_size`.
      epsilon: A floating number. A very small number for preventing ZeroDivision Error.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A tensor with the same shape and data dtype as `inputs`.
    '''
    with tf.variable_scope(scope, reuse=reuse):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]

        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta = tf.Variable(tf.zeros(params_shape))
        gamma = tf.Variable(tf.ones(params_shape))
        normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
        outputs = gamma * normalized + beta

    return outputs