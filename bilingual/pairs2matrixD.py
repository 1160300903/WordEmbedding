import numpy as np 
from corpus2vocab import read_vocab
from scipy.sparse import dok_matrix, csc_matrix
from matrix_sl import save_matrix
import setting as st
def form_matrix(text, src_word2index, trg_word2index, D):
    tmp_D = dok_matrix(D.shape,dtype="float32")
    times = 0
    for k in range(len(text)):
        src_word, trg_word, similarity = text[k]
        tmp_D[src_word2index[src_word], trg_word2index[trg_word]] = float(similarity)
        tmp_D[trg_word2index[trg_word], src_word2index[src_word]] = float(similarity)
        times += 1
        if times == st.UPDATE_THRESHOLD:
            D = D + tmp_D.tocsc()
            tmp_D = dok_matrix(D.shape,dtype="float32")
            times = 0
    D = D + tmp_D.tocsc()
    return D
def compute_D(pairs_file, src_vocab, trg_vocab, output_file):
    src_word2index = read_vocab(src_vocab)
    trg_word2index = read_vocab(trg_vocab)
    length = len(src_word2index)+len(trg_word2index)
    D = csc_matrix((length, length), dtype="float32")

    input = open(pairs_file, "r", encoding="UTF-8-sig")
    pairs = [line.strip().split() for line in input.readlines()]
    input.close()

    D = form_matrix(pairs, src_word2index, trg_word2index, D)
    save_matrix(output_file, D)

if __name__ == "__main__":
    pairs_file = st.PAIRS_DIR + ""
    src_vocab, trg_vocab = st.VOCAB_DIR + "", st.VOCAB_DIR + ""
    output_file = st.MATRIX_DIR + ""
    compute_D(pairs_file, src_vocab, trg_vocab, output_file)

    