import numpy as np
import numpy.linalg as LA
import gensim
from corpus2vocab import read_vocab
import setting as st
import sys

def align(word2index):
    min_index = 10000000000
    for word in word2index:
        min_index = min(min_index, word2index[word])
    if min_index > 0:
        for word in word2index:
            word2index[word] = word2index[word] - min_index

#since the embedding words set equals the context words set in my experiment,
#I just use embedding words to form D_1 and D_2
def vec2pairs(src_mono_vec, trg_mono_vec, src_vocab, trg_vocab):
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

    trg_index2word = {trg_word2index[word]:word for word in trg_word2index}

    trg_matrix = np.zeros((trg_vectors.vector_size, len(trg_word2index)))

    print("form the target language matrix")
    for word in trg_word2index:
        trg_matrix[:,trg_word2index[word]] = trg_vectors[word]

    # compute the norm-2 of every word vector. axis=0 means regarding every column as a vector
    trg_vec_length = np.reciprocal(LA.norm(trg_matrix, axis=0))
    for i in range(len(trg_vec_length)):
        trg_matrix[:, i] *= trg_vec_length[i]

    words = ['abdomen', 'magnesium', 'ammonia', 'yuwen','everton']
    for word in words:
        if word not in src_word2index:
            continue
        a = LA.norm(src_vectors[word])
        sim_mat = src_vectors[word].dot(trg_matrix)
        indices = np.argpartition(sim_mat, -100)[-100:]
        print(word)
        for idx in indices:
            print(trg_index2word[idx], end=" ")
        print()
        for idx in indices:
            print(sim_mat[idx]/a, end=" ")
        print()

if __name__ == "__main__":
    src_vocab = st.VOCAB_DIR + sys.argv[1] if len(sys.argv) > 1 else st.VOCAB_DIR + "F10-W5.2en"
    trg_vocab = st.VOCAB_DIR + sys.argv[2] if len(sys.argv) > 2 else st.VOCAB_DIR + "F10-W5.2zh"

    src_vec = sys.argv[3] if len(sys.argv) > 3 else "../../word2vec/mapped.2en"
    trg_vec = sys.argv[4] if len(sys.argv) > 4 else "../../word2vec/mapped.2zh"

    vec2pairs(src_vec, trg_vec, src_vocab, trg_vocab)

    
