import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell
import tensorflow_hub as hub
from model.attention import attention


class SelfAttentiveLSTM:
    def __init__(self, sequence_length, num_classes, vocab_size, embedding_size,
                 hidden_size, attention_size, use_elmo=False, l2_reg_lambda=0.0):
        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, shape=[None, sequence_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, shape=[None, num_classes], name='input_y')
        self.input_text = tf.placeholder(tf.string, shape=[None, ], name='input_text')
        self.rnn_dropout_keep_prob = tf.placeholder(tf.float32, name='rnn_dropout_keep_prob')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)
        text_length = self._length(self.input_x)

        if use_elmo:
            with tf.variable_scope("elmo-embeddings"):
                elmo_model = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)
                self.embedded_chars = elmo_model(self.input_text, signature="default", as_dict=True)["elmo"]
        else:
            # Embedding layer
            with tf.device('/cpu:0'), tf.variable_scope("word-embeddings"):
                self.W_text = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), name="W_text")
                self.embedded_chars = tf.nn.embedding_lookup(self.W_text, self.input_x)

        # with tf.variable_scope("self-attention"):
        #     self.entity_att = self.multihead_attention(self.embedded_chars,
        #                                                self.embedded_chars,
        #                                                self.embedded_chars,
        #                                                dim_model=embedding_size, num_head=3)

        with tf.variable_scope("bi-rnn"):
            fw_cell = tf.nn.rnn_cell.DropoutWrapper(LSTMCell(hidden_size), self.rnn_dropout_keep_prob)
            bw_cell = tf.nn.rnn_cell.DropoutWrapper(LSTMCell(hidden_size), self.rnn_dropout_keep_prob)
            self.rnn_outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_cell,
                                                                  cell_bw=bw_cell,
                                                                  inputs=self.embedded_chars,
                                                                  sequence_length=text_length,
                                                                  dtype=tf.float32)

        # Attention layer
        with tf.variable_scope('attention'):
            self.att_output, self.alphas = attention(self.rnn_outputs, attention_size, return_alphas=True)

        # Dropout
        with tf.variable_scope('dropout'):
            self.h_drop = tf.nn.dropout(self.att_output, self.dropout_keep_prob)

        # Fully connected layer
        with tf.variable_scope('output'):
            W = tf.get_variable("W", shape=[hidden_size*2, num_classes],
                                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.0, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.logits = tf.nn.xw_plus_b(self.h_drop, W, b, name="logits")
            self.predictions = tf.argmax(self.logits, 1, name="predictions")

        # Calculate mean cross-entropy loss
        with tf.variable_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.variable_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name="accuracy")

    @staticmethod
    def multihead_attention(query, key, value, dim_model, num_head):
        Q = tf.layers.dense(query, dim_model, activation=tf.nn.relu)
        K = tf.layers.dense(key, dim_model, activation=tf.nn.relu)
        V = tf.layers.dense(value, dim_model, activation=tf.nn.relu)

        # Scaled Dot Product Attention
        with tf.variable_scope("scaled-dot-product-attention"):
            QK_T = tf.matmul(Q, tf.transpose(K, [0, 2, 1]))
            att = tf.nn.softmax(QK_T * tf.sqrt(1/dim_model))
            att_V = tf.matmul(att, V)

        output = tf.layers.dense(att_V, dim_model, activation=tf.nn.relu)

        return output

    # Length of the sequence data
    @staticmethod
    def _length(seq):
        relevant = tf.sign(tf.abs(seq))
        length = tf.reduce_sum(relevant, reduction_indices=1)
        length = tf.cast(length, tf.int32)
        return length