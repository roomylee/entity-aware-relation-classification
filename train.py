import datetime
import os
import time
import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score

import data_helpers
import configure
from model.self_att_lstm import SelfAttentiveLSTM
import utils

import warnings
import sklearn.exceptions
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)


FLAGS = configure.parse_args()


def train():
    with tf.device('/cpu:0'):
        train_text, train_y = data_helpers.load_data_and_labels(FLAGS.train_path)
    with tf.device('/cpu:0'):
        test_text, test_y = data_helpers.load_data_and_labels(FLAGS.test_path)

    # Build vocabulary
    # Example: x_text[3] = "A misty <e1>ridge</e1> uprises from the <e2>surge</e2>."
    # ['a misty ridge uprises from the surge <UNK> <UNK> ... <UNK>']
    # =>
    # [27 39 40 41 42  1 43  0  0 ... 0]
    # dimension = MAX_SENTENCE_LENGTH
    if FLAGS.vocab_path:
        vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor.restore(FLAGS.vocab_path)
    else:
        vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(FLAGS.max_sentence_length)
        vocab_processor.fit(train_text)
    train_x = np.array(list(vocab_processor.transform(train_text)))
    test_x = np.array(list(vocab_processor.transform(test_text)))
    train_text = np.array(train_text)
    test_text = np.array(test_text)
    print("Text Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
    print("train_x = {0}".format(train_x.shape))
    print("train_y = {0}".format(train_y.shape))
    print("test_x = {0}".format(test_x.shape))
    print("test_y = {0}".format(test_y.shape))
    print("")

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        session_conf.gpu_options.allow_growth = FLAGS.gpu_allow_growth
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            model = SelfAttentiveLSTM(
                sequence_length=train_x.shape[1],
                num_classes=train_y.shape[1],
                vocab_size=len(vocab_processor.vocabulary_),
                embedding_size=FLAGS.embedding_size,
                hidden_size=FLAGS.hidden_size,
                attention_size=FLAGS.attention_size,
                use_elmo=FLAGS.elmo,
                l2_reg_lambda=FLAGS.l2_reg_lambda)

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            train_op = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(model.loss, global_step=global_step)

            # Output directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
            print("Writing to {}\n".format(out_dir))

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", model.loss)
            acc_summary = tf.summary.scalar("accuracy", model.accuracy)

            # Train Summaries
            train_summary_op = tf.summary.merge([loss_summary, acc_summary])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Dev summaries
            test_summary_op = tf.summary.merge([loss_summary, acc_summary])
            test_summary_dir = os.path.join(out_dir, "summaries", "dev")
            test_summary_writer = tf.summary.FileWriter(test_summary_dir, sess.graph)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

            # Write vocabulary
            vocab_processor.save(os.path.join(out_dir, "vocab"))
            utils.save_result(np.argmax(test_y, axis=1), os.path.join(out_dir, FLAGS.target_path), mkdir=True)

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            if not FLAGS.elmo and FLAGS.word2vec:
                pretrain_W = utils.load_word2vec(FLAGS.word2vec, FLAGS.embedding_size, vocab_processor)
                sess.run(model.W_text.assign(pretrain_W))
                print("Success to load pre-trained word2vec model!\n")

            # Generate batches
            train_batches = data_helpers.batch_iter(list(zip(train_x, train_y, train_text)),
                                              FLAGS.batch_size, FLAGS.num_epochs)
            # Training loop. For each batch...
            best_f1 = 0.0  # For save checkpoint(model)
            for train_batch in train_batches:
                train_bx, train_by, train_btxt = zip(*train_batch)
                feed_dict = {
                    model.input_x: train_bx,
                    model.input_y: train_by,
                    model.input_text: train_btxt,
                    model.rnn_dropout_keep_prob: FLAGS.rnn_dropout_keep_prob,
                    model.dropout_keep_prob: FLAGS.dropout_keep_prob
                }
                _, step, summaries, loss, accuracy = sess.run(
                    [train_op, global_step, train_summary_op, model.loss, model.accuracy], feed_dict)
                train_summary_writer.add_summary(summaries, step)

                # Training log display
                if step % FLAGS.display_every == 0:
                    time_str = datetime.datetime.now().isoformat()
                    print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))

                # Evaluation
                if step % FLAGS.evaluate_every == 0:
                    print("\nEvaluation:")
                    # Generate batches
                    test_batches = data_helpers.batch_iter(list(zip(test_x, test_y, test_text)),
                                                           FLAGS.batch_size, 1, shuffle=False)
                    # Training loop. For each batch...
                    losses = 0.0
                    accuracy = 0.0
                    predictions = []
                    for test_batch in test_batches:
                        test_bx, test_by, test_btxt = zip(*test_batch)
                        feed_dict = {
                            model.input_x: test_bx,
                            model.input_y: test_by,
                            model.input_text: test_btxt,
                            model.rnn_dropout_keep_prob: 1.0,
                            model.dropout_keep_prob: 1.0
                        }
                        summaries, loss, acc, pred = sess.run(
                            [test_summary_op, model.loss, model.accuracy, model.predictions], feed_dict)
                        test_summary_writer.add_summary(summaries, step)
                        losses += loss
                        accuracy += acc
                        predictions += pred

                    losses /= int(len(test_y) / FLAGS.batch_size)
                    accuracy /= int(len(test_y) / FLAGS.batch_size)
                    predictions = np.array(predictions, dtype='int')
                    f1 = f1_score(np.argmax(test_y, axis=1), predictions, labels=np.array(range(1, 19)), average="macro")

                    time_str = datetime.datetime.now().isoformat()
                    print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, losses, accuracy))
                    print("(2*9+1)-Way Macro-Average F1 Score (excluding Other): {:g}\n".format(f1))

                    # Model checkpoint
                    if best_f1 * 0.98 < f1:
                        if best_f1 < f1:
                            best_f1 = f1
                        path = saver.save(sess, checkpoint_prefix+"-{:.3g}".format(f1), global_step=step)
                        output_path = FLAGS.output_path[:-4]+"-{:.3g}-{}".format(f1, step)+".txt"
                        utils.save_result(predictions, os.path.join(out_dir, output_path))
                        print("Saved model checkpoint to {}\n".format(path))


def main(_):
    train()


if __name__ == "__main__":
    tf.app.run()
