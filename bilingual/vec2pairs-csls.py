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
#since the embedding words set equals the context words set in my experiment,
#I just use embedding words to form D_1 and D_2
def vec2pairs(src_mono_vec, trg_mono_vec, src_vocab, trg_vocab, output_file, TOP_TRANS):
    print("loading source vectors")
    src_vectors = gensim.models.KeyedVectors.load_word2vec_format(src_mono_vec, binary = False)
    print("loading target vectors")
    trg_vectors = gensim.models.KeyedVectors.load_word2vec_format(trg_mono_vec, binary = False)

    print("load src_word2index")
    src_word2index = read_vocab(src_vocab)
    print("load trg_word2index")
    trg_word2index = read_vocab(trg_vocab)
    length = len(src_word2index)
    trg_word2index = {word:trg_word2index[word]-length for word in trg_word2index}

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

    knn_sim_bwd = np.zeros(trg_matrix.shape[1])
    for i in range(0, trg_matrix.shape[1], batch):
        j = min(i + batch, trg_matrix.shape[1])
        knn_sim_bwd[i:j] = topk_mean(trg_matrix.T[i:j].dot(src_matrix.T), k=10, inplace=True)

    for i in range(0, src_matrix.shape[0], batch):
        print("batch",i/batch)
        temp = src_matrix[i: min(i + batch, src_matrix.shape[0]),:]
        similarity = 2 * np.dot(temp, trg_matrix) - knn_sim_bwd
        index = np.argpartition(similarity,-TOP_TRANS,axis=1)[:,-TOP_TRANS:]
        for j in range(0, temp.shape[0]):
            sum = 0
            for k in range(TOP_TRANS):
                sum += similarity[j, index[j, k]]
            for k in range(TOP_TRANS):
                output.write(src_index2word[i+j]+" "+trg_index2word[index[j, k]]+" "+str(similarity[j,index[j, k]]/sum)+"\n")
    output.close()
if __name__ == "__main__":
    TOP_TRANS = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    src_vocab = st.VOCAB_DIR + sys.argv[2] if len(sys.argv) > 2 else st.VOCAB_DIR + "F10-W5.1en"
    trg_vocab = st.VOCAB_DIR + sys.argv[3] if len(sys.argv) > 3 else st.VOCAB_DIR + "F10-W5.1de"

    src_mono_vec = st.RDM_VEC_DIR
    trg_mono_vec = st.RDM_VEC_DIR

    src_mono_vec += sys.argv[4] if len(sys.argv) > 4 else "F10-W5.1en"
    trg_mono_vec += sys.argv[5] if len(sys.argv) > 5 else "F10-W5.1de"

    param = src_mono_vec.split("/")[-1].split(".")[0]
    output_file = st.PAIRS_DIR + param + "-T" + str(TOP_TRANS) + "."
    output_file += sys.argv[6] if len(sys.argv) > 6 else "en-de"

    vec2pairs(src_mono_vec, trg_mono_vec, src_vocab, trg_vocab, output_file, TOP_TRANS)

    
