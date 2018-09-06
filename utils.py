import os
import numpy as np

labelsMapping = {0: 'Other',
                 1: 'Message-Topic(e1,e2)', 2: 'Message-Topic(e2,e1)',
                 3: 'Product-Producer(e1,e2)', 4: 'Product-Producer(e2,e1)',
                 5: 'Instrument-Agency(e1,e2)', 6: 'Instrument-Agency(e2,e1)',
                 7: 'Entity-Destination(e1,e2)', 8: 'Entity-Destination(e2,e1)',
                 9: 'Cause-Effect(e1,e2)', 10: 'Cause-Effect(e2,e1)',
                 11: 'Component-Whole(e1,e2)', 12: 'Component-Whole(e2,e1)',
                 13: 'Entity-Origin(e1,e2)', 14: 'Entity-Origin(e2,e1)',
                 15: 'Member-Collection(e1,e2)', 16: 'Member-Collection(e2,e1)',
                 17: 'Content-Container(e1,e2)', 18: 'Content-Container(e2,e1)'}


def load_word2vec(word2vec_path, embedding_dim, vocab):
    # initial matrix with random uniform
    initW = np.random.uniform(-1.0, 1.0, (len(vocab.vocabulary_), embedding_dim))
    # load any vectors from the word2vec
    print("Load word2vec file {0}".format(word2vec_path))
    with open(word2vec_path, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in range(vocab_size):
            word = []
            while True:
                ch = f.read(1).decode('latin-1')
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)
            idx = vocab.vocabulary_.get(word)
            if idx != 0:
                initW[idx] = np.fromstring(f.read(binary_len), dtype='float32')
            else:
                f.read(binary_len)
    return initW


def save_result(output, path, mkdir=False):
    if mkdir:
        os.makedirs(path[:-10])
    output_file = open(path, 'w')
    for i in range(len(output)):
        output_file.write("{}\t{}\n".format(i, labelsMapping[output[i]]))
    output_file.close()