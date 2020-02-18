from counts2matrixX import compute_X
from pairs2matrixD import compute_D
from corpus2vocab import read_vocab
import numpy as np
from sparsesvd import sparsesvd
from scipy.sparse import csc_matrix, eye
from matrix_sl import load_matrix
import setting as st

def svd(matrixX, matrixD, src_vocab, trg_vocab, src_vec, trg_vec):
    print("load word2index")
    src_word2index = read_vocab(src_vocab)
    trg_word2index = read_vocab(trg_vocab)
    print("load X")
    X = load_matrix(matrixX)
    print("load D")
    D = load_matrix(matrixD)
    I = eye(X.shape[0], format="csc")
    ID = I+D
    print("start svd")
    u = sparsesvd(X,ID, st.VECTOR_LENGTH)[0].T
    print("finish svd")
    print("output vectors")
    output = open(src_vec, "w",encoding="utf-8")
    output.write(str(len(src_word2index))+" "+str(st.VECTOR_LENGTH)+"\n")
    for word in src_word2index:
        vector = u[src_word2index[word]]
        output.write(word)
        for i in range(st.VECTOR_LENGTH):
            output.write(" %.8f"%vector[i])
        output.write("\n")
    output.close()

    output = open(trg_vec, "w",encoding="utf-8")
    output.write(str(len(trg_word2index))+" "+str(st.VECTOR_LENGTH)+"\n")
    for word in trg_word2index:
        vector = u[trg_word2index[word]]
        output.write(word)
        for i in range(st.VECTOR_LENGTH):
            output.write(" %.8f"%vector[i])
        output.write("\n")
    output.close()
if __name__ =="__main__":
    matrixX, matrixD = st.MATRIX_DIR + "", st.MATRIX_DIR + ""
    src_vocab, trg_vocab = st.VOCAB_DIR + "", st.VOCAB_DIR + ""
    src_vec, trg_vec = st.RES_DIR + "L" + str(st.VECTOR_LENGTH) + ".", st.RES_DIR + "L" + str(st.VECTOR_LENGTH) + "."
    svd(matrixX, matrixD, src_vocab, trg_vocab, src_vec, trg_vec)