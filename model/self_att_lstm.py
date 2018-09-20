import tensorflow as tf
import tensorflow_hub as hub
from model.attention import attention_with_latent_var, attention_with_no_size, multihead_attention, entity_attention, \
    relative_multihead_attention, latent_type_attention


class SelfAttentiveLSTM:
    def __init__(self, sequence_length, num_classes,
                 vocab_size, embedding_size, dist_vocab_size, dist_embedding_size,
                 hidden_size, clip_k, num_heads, attention_size,
                 use_elmo=False, l2_reg_lambda=0.0):
        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, shape=[None, sequence_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, shape=[None, num_classes], name='input_y')
        self.input_text = tf.placeholder(tf.string, shape=[None, ], name='input_text')
        self.input_e1 = tf.placeholder(tf.int32, shape=[None, ], name='input_e1')
        self.input_e2 = tf.placeholder(tf.int32, shape=[None, ], name='input_e2')
        self.input_d1 = tf.placeholder(tf.int32, shape=[None, sequence_length], name='input_d1')
        self.input_d2 = tf.placeholder(tf.int32, shape=[None, sequence_length], name='input_d2')
        self.emb_dropout_keep_prob = tf.placeholder(tf.float32, name='emb_dropout_keep_prob')
        self.rnn_dropout_keep_prob = tf.placeholder(tf.float32, name='rnn_dropout_keep_prob')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')

        initializer = tf.contrib.layers.xavier_initializer()
        text_length = self._length(self.input_x)

        if use_elmo:
            with tf.variable_scope("elmo-embeddings"):
                elmo_model = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)
                self.embedded_chars = elmo_model(self.input_text, signature="default", as_dict=True)["elmo"]
        else:
            # Embedding layer
            with tf.device('/cpu:0'), tf.variable_scope("word-embeddings"):
                self.W_text = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -0.25, 0.25), name="W_text")
                self.embedded_chars = tf.nn.embedding_lookup(self.W_text, self.input_x)

        # with tf.device('/cpu:0'), tf.variable_scope("position-embeddings"):
        #     self.W_dist = tf.get_variable("W_dist", [dist_vocab_size, dist_embedding_size], initializer=initializer)
        #     self.d1 = tf.nn.embedding_lookup(self.W_dist, self.input_d1)
        #     self.d2 = tf.nn.embedding_lookup(self.W_dist, self.input_d2)
        #
        # self.embedded_x = tf.concat([self.embedded_chars, self.d1, self.d2], axis=-1)

        with tf.variable_scope('dropout-embeddings'):
            self.embedded_chars = tf.nn.dropout(self.embedded_chars,  self.emb_dropout_keep_prob)

        with tf.variable_scope("self-attention"):
            # self.self_att = multihead_attention(self.embedded_chars, self.embedded_chars,
            #                                     num_units=embedding_size, num_heads=num_heads)
            self.self_att = relative_multihead_attention(self.embedded_chars, self.embedded_chars,
                                                         num_units=embedding_size, num_heads=num_heads,
                                                         clip_k=clip_k, seq_len=sequence_length)

        with tf.variable_scope("entity-attention"):
            self.entity_att = entity_attention(self.self_att, self.input_e1, self.input_e2,
                                               attention_size=attention_size)
            self.concat_att = tf.concat([self.self_att, self.entity_att], axis=-1)

        with tf.variable_scope("bi-rnn"):
            _fw_cell = tf.nn.rnn_cell.LSTMCell(hidden_size)
            fw_cell = tf.nn.rnn_cell.DropoutWrapper(_fw_cell, self.rnn_dropout_keep_prob)
            _bw_cell = tf.nn.rnn_cell.LSTMCell(hidden_size)
            bw_cell = tf.nn.rnn_cell.DropoutWrapper(_bw_cell, self.rnn_dropout_keep_prob)
            self.rnn_outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_cell,
                                                                  cell_bw=bw_cell,
                                                                  inputs=self.concat_att,
                                                                  sequence_length=text_length,
                                                                  dtype=tf.float32)
            self.rnn_outputs_concat = tf.concat(self.rnn_outputs, axis=-1)
            self.rnn_outputs_add = tf.add(self.rnn_outputs[0], self.rnn_outputs[1])

        # Attention layer
        with tf.variable_scope('attention-with-latent-var'):
            self.att_output, self.alphas = attention_with_latent_var(self.rnn_outputs_concat, attention_size)
        # with tf.variable_scope('attention-with-no-size'):
        #     self.att_output, self.alphas = attention_with_no_size(self.rnn_outputs_add)

        # Latent Entity Type
        with tf.variable_scope("latent-type-attention"):
            self.e1_type, self.e2_type = latent_type_attention(self.rnn_outputs_concat, self.input_e1, self.input_e2,
                                                               num_type=3, latent_size=attention_size)
            self.output_lt = tf.concat([self.att_output, self.e1_type, self.e2_type], axis=-1)

        # Dropout
        with tf.variable_scope('dropout'):
            self.h_drop = tf.nn.dropout(self.output_lt, self.dropout_keep_prob)

        with tf.variable_scope('batch-norm'):
            self.h_drop = tf.layers.batch_normalization(self.h_drop)

        # Fully connected layer
        with tf.variable_scope('output'):
            self.logits = tf.layers.dense(self.h_drop, num_classes, kernel_initializer=initializer)
            self.predictions = tf.argmax(self.logits, 1, name="predictions")

        # Calculate mean cross-entropy loss
        with tf.variable_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.input_y)
            # losses = self.ranking_loss(self.logits, self.input_y)
            self.l2 = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * self.l2

        # Accuracy
        with tf.variable_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name="accuracy")

    # Length of the sequence data
    @staticmethod
    def _length(seq):
        relevant = tf.sign(tf.abs(seq))
        length = tf.reduce_sum(relevant, reduction_indices=1)
        length = tf.cast(length, tf.int32)
        return length

    @staticmethod
    def ranking_loss(logits, input_y, gamma=1.0, margin_pos=2.5, margin_neg=-0.5):
        labels = tf.argmax(input_y, 1, output_type=tf.int32)

        gamma = tf.constant(gamma)  # lambda
        mp = tf.constant(margin_pos)
        mn = tf.constant(margin_neg)

        L = tf.constant(0.0)
        i = tf.constant(0)
        cond = lambda i, L: tf.less(i, tf.shape(labels)[0])

        def loop_body(i, L):
            pos_label = labels[i]  # positive class label index
            # taking most informative negative class, use 2nd argmax
            _, neg_indices = tf.nn.top_k(logits[i], k=2)
            max_neg_index = tf.cond(tf.equal(pos_label, neg_indices[0]),
                                    lambda: neg_indices[1], lambda: neg_indices[0])

            s_pos = logits[i, pos_label]  # score for gold class
            s_neg = logits[i, max_neg_index]  # score for negative class

            l = tf.log((1.0 + tf.exp((gamma * (mp - s_pos))))) + tf.log((1.0 + tf.exp((gamma * (mn + s_neg)))))

            return [tf.add(i, 1), tf.add(L, l)]

        _, L = tf.while_loop(cond, loop_body, loop_vars=[i, L])
        return L
