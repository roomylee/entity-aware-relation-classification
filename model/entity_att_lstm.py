import tensorflow as tf
import tensorflow_hub as hub

from utils import initializer
from model.attention import multihead_attention, attention


class EntityAttentionLSTM:
    def __init__(self, sequence_length, num_classes,
                 vocab_size, embedding_size, pos_vocab_size, pos_embedding_size,
                 hidden_size, num_heads, attention_size,
                 use_elmo=False, l2_reg_lambda=0.0):
        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, shape=[None, sequence_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, shape=[None, num_classes], name='input_y')
        self.input_text = tf.placeholder(tf.string, shape=[None, ], name='input_text')
        self.input_e1 = tf.placeholder(tf.int32, shape=[None, ], name='input_e1')
        self.input_e2 = tf.placeholder(tf.int32, shape=[None, ], name='input_e2')
        self.input_p1 = tf.placeholder(tf.int32, shape=[None, sequence_length], name='input_p1')
        self.input_p2 = tf.placeholder(tf.int32, shape=[None, sequence_length], name='input_p2')
        self.emb_dropout_keep_prob = tf.placeholder(tf.float32, name='emb_dropout_keep_prob')
        self.rnn_dropout_keep_prob = tf.placeholder(tf.float32, name='rnn_dropout_keep_prob')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')

        if use_elmo:
            # Contextual Embedding Layer
            with tf.variable_scope("elmo-embeddings"):
                elmo_model = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)
                self.embedded_chars = elmo_model(self.input_text, signature="default", as_dict=True)["elmo"]
        else:
            # Word Embedding Layer
            with tf.device('/cpu:0'), tf.variable_scope("word-embeddings"):
                self.W_text = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -0.25, 0.25), name="W_text")
                self.embedded_chars = tf.nn.embedding_lookup(self.W_text, self.input_x)

        # Position Embedding Layer
        with tf.device('/cpu:0'), tf.variable_scope("position-embeddings"):
            self.W_pos = tf.get_variable("W_pos", [pos_vocab_size, pos_embedding_size], initializer=initializer())
            self.p1 = tf.nn.embedding_lookup(self.W_pos, self.input_p1)[:, :tf.shape(self.embedded_chars)[1]]
            self.p2 = tf.nn.embedding_lookup(self.W_pos, self.input_p2)[:, :tf.shape(self.embedded_chars)[1]]

        # Dropout for Word Embedding
        with tf.variable_scope('dropout-embeddings'):
            self.embedded_chars = tf.nn.dropout(self.embedded_chars,  self.emb_dropout_keep_prob)

        # Self Attention
        with tf.variable_scope("self-attention"):
            self.self_attn, self.self_alphas = multihead_attention(self.embedded_chars, self.embedded_chars,
                                                                   num_units=embedding_size, num_heads=num_heads)

        # Bidirectional LSTM
        with tf.variable_scope("bi-lstm"):
            _fw_cell = tf.nn.rnn_cell.LSTMCell(hidden_size, initializer=initializer())
            fw_cell = tf.nn.rnn_cell.DropoutWrapper(_fw_cell, self.rnn_dropout_keep_prob)
            _bw_cell = tf.nn.rnn_cell.LSTMCell(hidden_size, initializer=initializer())
            bw_cell = tf.nn.rnn_cell.DropoutWrapper(_bw_cell, self.rnn_dropout_keep_prob)
            self.rnn_outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_cell,
                                                                  cell_bw=bw_cell,
                                                                  inputs=self.self_attn,
                                                                  sequence_length=self._length(self.input_x),
                                                                  dtype=tf.float32)
            self.rnn_outputs = tf.concat(self.rnn_outputs, axis=-1)

        # Attention
        with tf.variable_scope('attention'):
            self.attn, self.alphas, self.e1_alphas, self.e2_alphas = attention(self.rnn_outputs,
                                                                               self.input_e1, self.input_e2,
                                                                               self.p1, self.p2,
                                                                               attention_size=attention_size)

        # Dropout
        with tf.variable_scope('dropout'):
            self.h_drop = tf.nn.dropout(self.attn, self.dropout_keep_prob)

        # Fully connected layer
        with tf.variable_scope('output'):
            self.logits = tf.layers.dense(self.h_drop, num_classes, kernel_initializer=initializer())
            self.predictions = tf.argmax(self.logits, 1, name="predictions")

        # Calculate mean cross-entropy loss
        with tf.variable_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.input_y)
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
