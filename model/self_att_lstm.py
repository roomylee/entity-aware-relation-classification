import tensorflow as tf
import tensorflow_hub as hub
from model.attention import *
from configure import FLAGS
from model.char_cnn import highway_fc_layer, char_cnn


class SelfAttentiveLSTM:
    def __init__(self, sequence_length, word_length, num_classes, vocab_size, embedding_size, dist_vocab_size, dist_embedding_size,
                 char_vocab_size, char_embedding_size, filter_sizes, num_filters,
                 hidden_size, attention_size, use_elmo=False, l2_reg_lambda=0.0):
        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, shape=[None, sequence_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, shape=[None, num_classes], name='input_y')
        self.input_char = tf.placeholder(tf.int32, shape=[None, sequence_length, word_length], name='input_char')
        self.input_text = tf.placeholder(tf.string, shape=[None, ], name='input_text')
        self.input_e1 = tf.placeholder(tf.int32, shape=[None, ], name='input_e1')
        self.input_e2 = tf.placeholder(tf.int32, shape=[None, ], name='input_e2')
        self.input_d1 = tf.placeholder(tf.int32, shape=[None, sequence_length], name='input_d1')
        self.input_d2 = tf.placeholder(tf.int32, shape=[None, sequence_length], name='input_d2')
        self.rnn_dropout_keep_prob = tf.placeholder(tf.float32, name='rnn_dropout_keep_prob')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')

        initializer = tf.contrib.layers.xavier_initializer()
        text_length = self._length(self.input_x)
        batch_size = tf.shape(self.input_x)[0]

        if use_elmo:
            with tf.variable_scope("elmo-embeddings"):
                elmo_model = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)
                self.embedded_chars = elmo_model(self.input_text, signature="default", as_dict=True)["elmo"]
        else:
            # Embedding layer
            with tf.device('/cpu:0'), tf.variable_scope("word-embeddings"):
                self.W_text = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -0.25, 0.25), name="W_text")
                self.embedded_chars = tf.nn.embedding_lookup(self.W_text, self.input_x)
                self.emb_e1 = tf.nn.embedding_lookup(self.W_text, self.input_e1)
                self.emb_e2 = tf.nn.embedding_lookup(self.W_text, self.input_e2)

        # with tf.device('/cpu:0'), tf.variable_scope("position-embeddings"):
        #     self.W_dist = tf.get_variable("W_dist", [dist_vocab_size, dist_embedding_size], initializer=initializer)
        #     self.d1 = tf.nn.embedding_lookup(self.W_dist, self.input_d1)
        #     self.d2 = tf.nn.embedding_lookup(self.W_dist, self.input_d2)
        #
        # self.embedded_x = tf.concat([self.embedded_chars, self.d1, self.d2], axis=-1)

        # with tf.variable_scope("char-embeddings"):
        #     self.W_char = tf.get_variable("W_char", [char_vocab_size, char_embedding_size], initializer=initializer)
        #     self.char_embedded_chars = tf.nn.embedding_lookup(self.W_char, self.input_char)
        #
        #     self.char_emb = char_cnn(self.char_embedded_chars,
        #                              filter_sizes=filter_sizes,
        #                              num_filters=num_filters,
        #                              char_embedding_size=char_embedding_size,
        #                              word_length=word_length,
        #                              sequence_length=sequence_length)
        #     num_filters_total = len(filter_sizes)*num_filters
        #     self.char_emb = highway_fc_layer(self.char_emb, num_filters_total)
        #
        # with tf.variable_scope("word-repr"):
        #     self.word_repr = tf.concat([self.embedded_chars, self.char_emb], axis=-1)

        with tf.variable_scope("self-attention"):
            self.entity_att = multihead_attention(self.embedded_chars, self.embedded_chars,
                                                  num_units=embedding_size,
                                                  num_heads=5)

        with tf.variable_scope("bi-rnn"):
            _fw_cell = tf.nn.rnn_cell.GRUCell(hidden_size)
            fw_cell = tf.nn.rnn_cell.DropoutWrapper(_fw_cell, self.rnn_dropout_keep_prob)
            fw_init = _fw_cell.zero_state(batch_size, tf.float32)
            _bw_cell = tf.nn.rnn_cell.GRUCell(hidden_size)
            bw_cell = tf.nn.rnn_cell.DropoutWrapper(_bw_cell, self.rnn_dropout_keep_prob)
            bw_init = _bw_cell.zero_state(batch_size, tf.float32)
            self.rnn_outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_cell,
                                                                  cell_bw=bw_cell,
                                                                  inputs=self.entity_att,
                                                                  sequence_length=text_length,
                                                                  initial_state_fw=fw_init,
                                                                  initial_state_bw=bw_init,
                                                                  dtype=tf.float32)

        # Attention layer
        with tf.variable_scope('attention'):
            self.att_output, self.alphas = attention(self.rnn_outputs, attention_size, return_alphas=True)

        # Dropout
        with tf.variable_scope('dropout'):
            self.h_drop = tf.nn.dropout(self.att_output, self.dropout_keep_prob)

        # with tf.variable_scope('concat'):
        #     E = tf.layers.dense(tf.concat([self.emb_e1, self.emb_e2], axis=-1), 100,
        #                         activation=tf.tanh, kernel_initializer=initializer)
        #     self.concat_output = tf.concat([self.h_drop, E], axis=-1)

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
