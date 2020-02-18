from counts2matrixX import compute_X
from pairs2matrixD import compute_D
from corpus2vocab import read_vocab
import numpy as np
from sparsesvd import sparsesvd
from scipy.sparse import csc_matrix, eye
from matrix_sl import load_matrix
import setting as st
import sys
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
    matrixX = st.MATRIX_DIR + sys.argv[1] if len(sys.argv) > 1 else st.MATRIX_DIR + ""
    matrixD = st.MATRIX_DIR + sys.argv[2] if len(sys.argv) > 2 else st.MATRIX_DIR + ""

    src_vocab = st.VOCAB_DIR + sys.argv[3] if len(sys.argv) > 3 else st.VOCAB_DIR + ""
    trg_vocab = st.VOCAB_DIR + sys.argv[4] if len(sys.argv) > 4 else st.VOCAB_DIR + ""
    
    para = matrixX.split("/")[-1].split(".")[0]

    src_vec = st.RES_DIR + para + "-L" + str(st.VECTOR_LENGTH) + "."
    src_vec += sys.argv[5] if len(sys.argv) > 5 else ""

    trg_vec = st.RES_DIR + para + "-L" + str(st.VECTOR_LENGTH) + "."
    trg_vec += sys.argv[6] if len(sys.argv) > 6 else ""

    svd(matrixX, matrixD, src_vocab, trg_vocab, src_vec, trg_vec)