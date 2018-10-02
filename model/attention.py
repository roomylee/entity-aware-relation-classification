import tensorflow as tf

from utils import initializer


def attention(inputs, e1, e2, p1, p2, attention_size):
    # inputs = (batch, seq_len, hidden)
    # e1, e2 = (batch, seq_len)
    # p1, p2 = (batch, seq_len, dist_emb_size)
    # attention_size = scalar(int)
    def extract_entity(x, e):
        e_idx = tf.concat([tf.expand_dims(tf.range(tf.shape(e)[0]), axis=-1), tf.expand_dims(e, axis=-1)], axis=-1)
        return tf.gather_nd(x, e_idx)  # (batch, hidden)
    seq_len = tf.shape(inputs)[1]  # fixed at run-time
    hidden_size = inputs.shape[2].value  # fixed at compile-time
    latent_size = hidden_size

    # Latent Relation Variable based on Entities
    e1_h = extract_entity(inputs, e1)  # (batch, hidden)
    e2_h = extract_entity(inputs, e2)  # (batch, hidden)
    e1_type, e2_type, e1_alphas, e2_alphas = latent_type_attention(e1_h, e2_h,
                                                                   num_type=3,
                                                                   latent_size=latent_size)  # (batch, hidden)
    e1_h = tf.concat([e1_h, e1_type], axis=-1)  # (batch, hidden+latent)
    e2_h = tf.concat([e2_h, e2_type], axis=-1)  # (batch, hidden+latent)

    # v*tanh(W*[h;p1;p2]+W*[e1;e2]) 85.18%? 84.83% 84.55%
    e_h = tf.layers.dense(tf.concat([e1_h, e2_h], -1), attention_size, use_bias=False, kernel_initializer=initializer())
    e_h = tf.reshape(tf.tile(e_h, [1, seq_len]), [-1, seq_len, attention_size])
    v = tf.layers.dense(tf.concat([inputs, p1, p2], axis=-1), attention_size, use_bias=False, kernel_initializer=initializer())
    v = tf.tanh(tf.add(v, e_h))

    u_omega = tf.get_variable("u_omega", [attention_size], initializer=initializer())
    vu = tf.tensordot(v, u_omega, axes=1, name='vu')  # (batch, seq_len)
    alphas = tf.nn.softmax(vu, name='alphas')  # (batch, seq_len)

    # v*tanh(W*[h;p1;p2;e1;e2]) 85.18% 84.41%
    # e1_h = tf.reshape(tf.tile(e1_h, [1, seq_len]), [-1, seq_len, hidden_size+latent_size])
    # e2_h = tf.reshape(tf.tile(e2_h, [1, seq_len]), [-1, seq_len, hidden_size+latent_size])
    # v = tf.concat([inputs, p1, p2, e1_h, e2_h], axis=-1)
    # v = tf.layers.dense(v, attention_size, activation=tf.tanh, kernel_initializer=initializer())
    #
    # u_omega = tf.get_variable("u_omega", [attention_size], initializer=initializer())
    # vu = tf.tensordot(v, u_omega, axes=1, name='vu')  # (batch, seq_len)
    # alphas = tf.nn.softmax(vu, name='alphas')  # (batch, seq_len)

    # output
    output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)  # (batch, hidden)

    return output, alphas, e1_alphas, e2_alphas


def latent_type_attention(e1, e2, num_type, latent_size):
    # Latent Entity Type Vectors
    latent_type = tf.get_variable("latent_type", shape=[num_type, latent_size], initializer=initializer())

    # e1_h = tf.layers.dense(e1, latent_size, kernel_initializer=initializer())
    # e2_h = tf.layers.dense(e2, latent_size, kernel_initializer=initializer())

    e1_sim = tf.matmul(e1, tf.transpose(latent_type))  # (batch, num_type)
    e1_alphas = tf.nn.softmax(e1_sim, name='e1_alphas')  # (batch, num_type)
    e1_type = tf.matmul(e1_alphas, latent_type, name='e1_type')  # (batch, hidden)

    e2_sim = tf.matmul(e2, tf.transpose(latent_type))  # (batch, num_type)
    e2_alphas = tf.nn.softmax(e2_sim, name='e2_alphas')  # (batch, num_type)
    e2_type = tf.matmul(e2_alphas, latent_type, name='e2_type')  # (batch, hidden)

    return e1_type, e2_type, e1_alphas, e2_alphas


def multihead_attention(queries, keys, num_units, num_heads,
                        dropout_rate=0, scope="multihead_attention", reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        # Linear projections
        Q = tf.layers.dense(queries, num_units, kernel_initializer=initializer())  # (N, T_q, C)
        K = tf.layers.dense(keys, num_units, kernel_initializer=initializer())  # (N, T_k, C)
        V = tf.layers.dense(keys, num_units, kernel_initializer=initializer())  # (N, T_k, C)

        # Split and concat
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, C/h)
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)

        # Multiplication
        outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # (h*N, T_q, T_k)

        # Scale
        outputs /= K_.get_shape().as_list()[-1] ** 0.5

        # Key Masking
        key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1)))  # (N, T_k)
        key_masks = tf.tile(key_masks, [num_heads, 1])  # (h*N, T_k)
        key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1])  # (h*N, T_q, T_k)

        paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
        outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

        # Activation
        alphas = tf.nn.softmax(outputs)  # (h*N, T_q, T_k)

        # Query Masking
        query_masks = tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1)))  # (N, T_q)
        query_masks = tf.tile(query_masks, [num_heads, 1])  # (h*N, T_q)
        query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]])  # (h*N, T_q, T_k)
        alphas *= query_masks  # broadcasting. (N, T_q, C)

        # Dropouts
        alphas = tf.layers.dropout(alphas, rate=dropout_rate, training=tf.convert_to_tensor(True))

        # Weighted sum
        outputs = tf.matmul(alphas, V_)  # ( h*N, T_q, C/h)

        # Restore shape
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # (N, T_q, C)

        # Linear
        outputs = tf.layers.dense(outputs, num_units, activation=tf.nn.relu, kernel_initializer=initializer())

        # Residual connection
        outputs += queries

        # Normalize
        outputs = layer_norm(outputs)  # (N, T_q, C)

    return outputs, alphas


def layer_norm(inputs, epsilon=1e-8, scope="layer_norm", reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]

        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta = tf.Variable(tf.zeros(params_shape))
        gamma = tf.Variable(tf.ones(params_shape))
        normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
        outputs = gamma * normalized + beta

    return outputs
