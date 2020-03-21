import numpy as np
import numpy.linalg as LA
import gensim
from corpus2vocab import read_vocab
import setting as st
import sys

# since the embedding words set equals the context words set in my experiment,
# I just use embedding words to form D_1 and D_2
def vec2pairs(src_mono_vec, trg_mono_vec, src_vocab, trg_vocab, output_file, TOP_TRANS):
    print("loading source vectors")
    src_vectors = gensim.models.KeyedVectors.load_word2vec_format(src_mono_vec, binary=False)
    print("loading target vectors")
    trg_vectors = gensim.models.KeyedVectors.load_word2vec_format(trg_mono_vec, binary=False)

    print("load src_word2index")
    src_word2index = read_vocab(src_vocab)
    print("load trg_word2index")
    trg_word2index = read_vocab(trg_vocab)
    length = len(src_word2index)
    trg_word2index = {word: trg_word2index[word] - length for word in trg_word2index}

    src_index2word = {src_word2index[word]: word for word in src_word2index}
    trg_index2word = {trg_word2index[word]: word for word in trg_word2index}
    top = 40000
    trg_matrix = np.zeros((trg_vectors.vector_size, top))
    src_matrix = np.zeros((top, src_vectors.vector_size))

    print("form the target language matrix")
    for word in trg_word2index:
        if trg_word2index[word] < top:
            trg_matrix[:, trg_word2index[word]] = trg_vectors[word]

    # compute the norm-2 of every word vector. axis=0 means regarding every column as a vector
    trg_vec_length = np.reciprocal(LA.norm(trg_matrix, axis=0))
    for i in range(len(trg_vec_length)):
        trg_matrix[:, i] *= trg_vec_length[i]

    print("form the source language matrix")
    for word in src_word2index:
        if src_word2index[word] < top:
            src_matrix[src_word2index[word], :] = src_vectors[word]

    # compute the norm-2 of every word vector
    src_vec_length = np.reciprocal(LA.norm(src_matrix, axis=1))
    for i in range(len(src_vec_length)):
        src_matrix[i, :] *= src_vec_length[i]

    print("finding translation pairs")
    batch = 1000
    output = open(output_file, "w", encoding="utf-8")
    for i in range(int(top / batch)):
        print("batch", i)
        temp = src_matrix[i * batch:i * batch + batch, :]
        result = np.dot(temp, trg_matrix)
        index = np.argpartition(result, -TOP_TRANS, axis=1)[:, -TOP_TRANS:]
        base = i * batch
        for j in range(0, batch):
            sum = 0
            for k in range(TOP_TRANS):
                sum += result[j, index[j, k]]
            for k in range(TOP_TRANS):
                output.write(src_index2word[base + j] + " " + trg_index2word[index[j, k]] + " " + str(
                    result[j, index[j, k]] / sum) + "\n")

    i = int(top / batch)
    temp = src_matrix[i * batch:top, :]
    result = np.dot(temp, trg_matrix)
    index = np.argpartition(result, -TOP_TRANS, axis=1)[:, -TOP_TRANS:]
    base = i * batch
    for j in range(0, temp.shape[0]):
        sum = 0
        for k in range(TOP_TRANS):
            sum += result[j, index[j, k]]
        for k in range(TOP_TRANS):
            output.write(src_index2word[base + j] + " " + trg_index2word[index[j, k]] + " " + str(
                result[j, index[j, k]] / sum) + "\n")
    output.close()


if __name__ == "__main__":
    src_vocab = st.VOCAB_DIR + sys.argv[1] if len(sys.argv) > 1 else st.VOCAB_DIR + "F1-W5.en"
    trg_vocab = st.VOCAB_DIR + sys.argv[2] if len(sys.argv) > 2 else st.VOCAB_DIR + "F1-W5.de"

    src_mono_vec = st.RDM_VEC_DIR
    trg_mono_vec = st.RDM_VEC_DIR

    src_mono_vec += sys.argv[3] if len(sys.argv) > 3 else "en"
    trg_mono_vec += sys.argv[4] if len(sys.argv) > 4 else "de"

    TOP_TRANS = 50

    param = src_mono_vec.split("/")[-1].split(".")[0]
    output_file = st.PAIRS_DIR + param + "-T" + str(TOP_TRANS) + "."
    output_file += sys.argv[5] if len(sys.argv) > 5 else "en-de"

    vec2pairs(src_mono_vec, trg_mono_vec, src_vocab, trg_vocab, output_file, TOP_TRANS)

