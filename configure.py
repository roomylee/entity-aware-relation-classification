import argparse
import sys


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser()

    # Data loading params
    parser.add_argument("--train_path", default="SemEval2010_task8_all_data/SemEval2010_task8_training/TRAIN_FILE.TXT",
                        type=str, help="Path of train data")
    parser.add_argument("--test_path", default="SemEval2010_task8_all_data/SemEval2010_task8_testing_keys/TEST_FILE_FULL.TXT",
                        type=str, help="Path of test data")
    parser.add_argument("--max_sentence_length", default=90,
                        type=int, help="Max sentence length in data")

    # Model Hyper-parameters
    # Embeddings
    parser.add_argument("--embeddings", default=None,
                        type=str, help="Embeddings {'word2vec', 'glove100', 'glove300', 'elmo'}")
    parser.add_argument("--embedding_size", default=300,
                        type=int, help="Dimensionality of word embedding (default: 300)")
    parser.add_argument("--dist_embedding_size", default=50,
                        type=int, help="Dimensionality of relative distance embedding (default: 50)")
    parser.add_argument("--emb_dropout_keep_prob", default=0.8,
                        type=float, help="Dropout keep probability of embedding layer (default: 0.8)")
    # RNN
    parser.add_argument("--hidden_size", default=300,
                        type=int, help="Dimensionality of RNN hidden (default: 512)")
    parser.add_argument("--rnn_dropout_keep_prob", default=0.7,
                        type=float, help="Dropout keep probability of RNN (default: 0.7)")
    # Attention
    parser.add_argument("--num_heads", default=4,
                        type=int, help="Number of heads in multi-head attention (default: 4)")
    parser.add_argument("--clip_k", default=3,
                        type=int, help="Size of clipping the relative position (default: 3)")
    parser.add_argument("--attention_size", default=50,
                        type=int, help="Dimensionality of attention (default: 50)")
    # Misc
    parser.add_argument("--desc", default="",
                        type=str, help="Description for model")
    parser.add_argument("--dropout_keep_prob", default=0.5,
                        type=float, help="Dropout keep probability of output layer (default: 0.5)")
    parser.add_argument("--l2_reg_lambda", default=1e-5,
                        type=float, help="L2 regularization lambda (default: 1e-5)")

    # Training parameters
    parser.add_argument("--batch_size", default=20,
                        type=int, help="Batch Size (default: 20)")
    parser.add_argument("--num_epochs", default=100,
                        type=int, help="Number of training epochs (Default: 100)")
    parser.add_argument("--display_every", default=10,
                        type=int, help="Number of iterations to display training information")
    parser.add_argument("--evaluate_every", default=100,
                        type=int, help="Evaluate model on dev set after this many steps (default: 100)")
    parser.add_argument("--num_checkpoints", default=5,
                        type=int, help="Number of checkpoints to store (default: 5)")
    parser.add_argument("--learning_rate", default=1e-3,
                        type=float, help="Which learning rate to start with (Default: 1e-3)")

    # Misc Parameters
    parser.add_argument("--allow_soft_placement", default=True,
                        type=bool, help="Allow device soft device placement")
    parser.add_argument("--log_device_placement", default=False,
                        type=bool, help="Log placement of ops on devices")
    parser.add_argument("--gpu_allow_growth", default=True,
                        type=bool, help="Allow gpu memory growth")

    if len(sys.argv) == 0:
        parser.print_help()
        sys.exit(1)

    print("")
    args = parser.parse_args()
    for arg in vars(args):
        print("{}={}".format(arg.upper(), getattr(args, arg)))
    print("")

    return args

FLAGS = parse_args()
