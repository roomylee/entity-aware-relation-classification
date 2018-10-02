import os
import numpy as np
import tensorflow as tf
import data_helpers
import logger
from configure import FLAGS
import warnings
import sklearn.exceptions

warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)

from tensor2tensor.visualization import attention


def visualize():
    with tf.device('/cpu:0'):
        test_text, test_y, test_e1, test_e2, test_pos1, test_pos2 = data_helpers.load_data_and_labels(FLAGS.test_path)

    checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
    print(checkpoint_file)

    vocab_path = os.path.join(FLAGS.checkpoint_dir, "..", "vocab")
    vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor.restore(vocab_path)

    # Map data into position
    position_path = os.path.join(FLAGS.checkpoint_dir, "..", "pos_vocab")
    pos_vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor.restore(position_path)

    test_x = np.array(list(vocab_processor.transform(test_text)))
    test_text = np.array(test_text)
    print("\nText Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
    print("test_x = {0}".format(test_x.shape))
    print("test_y = {0}".format(test_y.shape))

    test_p1 = np.array(list(pos_vocab_processor.transform(test_pos1)))
    test_p2 = np.array(list(pos_vocab_processor.transform(test_pos2)))
    print("\nPosition Vocabulary Size: {:d}".format(len(pos_vocab_processor.vocabulary_)))
    print("test_p1 = {0}".format(test_p1.shape))
    print("")

    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        session_conf.gpu_options.allow_growth = FLAGS.gpu_allow_growth
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            # Load the saved meta graph and restore variables
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            input_x = graph.get_operation_by_name("input_x").outputs[0]
            input_y = graph.get_operation_by_name("input_y").outputs[0]
            input_text = graph.get_operation_by_name("input_text").outputs[0]
            input_e1 = graph.get_operation_by_name("input_e1").outputs[0]
            input_e2 = graph.get_operation_by_name("input_e2").outputs[0]
            input_p1 = graph.get_operation_by_name("input_p1").outputs[0]
            input_p2 = graph.get_operation_by_name("input_p2").outputs[0]
            emb_dropout_keep_prob = graph.get_operation_by_name("emb_dropout_keep_prob").outputs[0]
            rnn_dropout_keep_prob = graph.get_operation_by_name("rnn_dropout_keep_prob").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
            self_alphas_op = graph.get_operation_by_name("self-attention/multihead_attention/Softmax").outputs[0]
            alphas_op = graph.get_operation_by_name("attention/alphas").outputs[0]
            acc_op = graph.get_operation_by_name("accuracy/accuracy").outputs[0]
            e2_alphas_op = graph.get_operation_by_name("attention/e2_alphas").outputs[0]
            e1_alphas_op = graph.get_operation_by_name("attention/e1_alphas").outputs[0]
            latent_type_op = graph.get_operation_by_name("attention/latent_type").outputs[0]

            print("\nEvaluation:")
            # Generate batches
            test_batches = data_helpers.batch_iter(list(zip(test_x, test_y, test_text,
                                                            test_e1, test_e2, test_p1, test_p2)),
                                                   FLAGS.batch_size, 1, shuffle=False)
            # Training loop. For each batch...
            accuracy = 0.0
            iter_cnt = 0
            with open("visualization.html", "w") as html_file:
                for test_batch in test_batches:
                    test_bx, test_by, test_btxt, test_be1, test_be2, test_bp1, test_bp2 = zip(*test_batch)
                    feed_dict = {
                        input_x: test_bx,
                        input_y: test_by,
                        input_text: test_btxt,
                        input_e1: test_be1,
                        input_e2: test_be2,
                        input_p1: test_bp1,
                        input_p2: test_bp2,
                        emb_dropout_keep_prob: 1.0,
                        rnn_dropout_keep_prob: 1.0,
                        dropout_keep_prob: 1.0
                    }
                    self_alphas, alphas, acc, e1_alphas, e2_alphas, latent_type = sess.run(
                        [self_alphas_op, alphas_op, acc_op, e1_alphas_op, e2_alphas_op, latent_type_op], feed_dict)
                    accuracy += acc
                    iter_cnt += 1
                    for text, alphas_values in zip(test_btxt, alphas):
                        for word, alpha in zip(text.split(), alphas_values / alphas_values.max()):
                            html_file.write(
                                '<font style="background: rgba(255, 255, 0, %f)">%s</font>\n' % (alpha, word))
                        html_file.write('<br>')
            accuracy /= iter_cnt
            print(accuracy)


def main(_):
    visualize()


if __name__ == "__main__":
    tf.app.run()
