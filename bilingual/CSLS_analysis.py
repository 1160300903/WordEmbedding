import numpy as np
import numpy.linalg as LA
import gensim
from corpus2vocab import read_vocab
import setting as st
import sys

def topk_mean(m, k, inplace=False):  # TODO Assuming that axis is 1
    n = m.shape[0]
    ans = np.zeros(n, dtype=m.dtype)
    if k <= 0:
        return ans
    if not inplace:
        m = np.array(m)
    ind0 = np.arange(n)
    ind1 = np.empty(n, dtype=int)
    minimum = m.min()
    for i in range(k):
        m.argmax(axis=1, out=ind1)
        ans += m[ind0, ind1]
        m[ind0, ind1] = minimum
    return ans / k

def align(word2index):
    min_index = 10000000000
    for word in word2index:
        min_index = min(min_index, word2index[word])
    if min_index > 0:
        for word in word2index:
            word2index[word] = word2index[word] - min_index

#since the embedding words set equals the context words set in my experiment,
#I just use embedding words to form D_1 and D_2
def vec2pairs(src_mono_vec, trg_mono_vec, src_vocab, trg_vocab, output_file):
    print("loading source vectors")
    src_vectors = gensim.models.KeyedVectors.load_word2vec_format(src_mono_vec, binary = False)
    print("loading target vectors")
    trg_vectors = gensim.models.KeyedVectors.load_word2vec_format(trg_mono_vec, binary = False)

    print("load src_word2index")
    src_word2index = read_vocab(src_vocab)
    print("load trg_word2index")
    trg_word2index = read_vocab(trg_vocab)
    align(src_word2index)
    align(trg_word2index)
    for word in src_word2index:
        print(src_word2index[word])

    src_index2word = {src_word2index[word]:word for word in src_word2index}
    trg_index2word = {trg_word2index[word]:word for word in trg_word2index}

    trg_matrix = np.zeros((trg_vectors.vector_size, len(trg_word2index)))
    src_matrix = np.zeros((len(src_word2index), src_vectors.vector_size))

    print("form the target language matrix")
    for word in trg_word2index:
        trg_matrix[:,trg_word2index[word]] = trg_vectors[word]

    # compute the norm-2 of every word vector. axis=0 means regarding every column as a vector
    trg_vec_length = np.reciprocal(LA.norm(trg_matrix, axis=0))
    for i in range(len(trg_vec_length)):
        trg_matrix[:, i] *= trg_vec_length[i]

    print("form the source language matrix")
    for word in src_word2index:
        src_matrix[src_word2index[word],:] = src_vectors[word]
    
    # compute the norm-2 of every word vector
    src_vec_length = np.reciprocal(LA.norm(src_matrix, axis=1))
    for i in range(len(src_vec_length)):
        src_matrix[i, :] *= src_vec_length[i]

    print("finding translation pairs")
    batch = 1000
    output = open(output_file,"w",encoding="utf-8")

    knn_sim_bwd = np.zeros(src_matrix.shape[0])# src_matrix.shape[0]是目标预言的单词个数
    for i in range(0, src_matrix.shape[0], batch):
        print("batch", i)
        j = min(i + batch, src_matrix.shape[0])
        knn_sim_bwd[i:j] = topk_mean(src_matrix[i:j].dot(trg_matrix), k=100, inplace=True)

    for i in range(0, knn_sim_bwd.shape[0]):
        output.write(src_index2word[i] + "," + str(knn_sim_bwd[i]) + "\n")

    output.close()
if __name__ == "__main__":
    src_vocab = st.VOCAB_DIR + sys.argv[1] if len(sys.argv) > 1 else st.VOCAB_DIR + "F10-W5.2zh"
    trg_vocab = st.VOCAB_DIR + sys.argv[2] if len(sys.argv) > 2 else st.VOCAB_DIR + "F10-W5.2zh"

    src_vec = sys.argv[3] if len(sys.argv) > 3 else "../../word2vec/mapped.2zh"
    trg_vec = sys.argv[4] if len(sys.argv) > 4 else "../../word2vec/mapped.2zh"

    output_file = sys.argv[6] if len(sys.argv) > 6 else "zh-zh.csv"

    vec2pairs(src_vec, trg_vec, src_vocab, trg_vocab, output_file)

    
