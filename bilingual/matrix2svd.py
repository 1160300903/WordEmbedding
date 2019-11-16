from counts2matrixX import compute_X
from pairs2matrixD import compute_D
from counts2vocab import read_vocab
import numpy as np
from sparsesvd import sparsesvd
from scipy.sparse import csc_matrix, eye
from matrix_sl import load_matrix
VECTOR_LENGTH = 300
def svd():
    print("load word2index")
    en_word2index = read_vocab("tempdata/en_word2index.txt")
    de_word2index = read_vocab("tempdata/de_word2index.txt")
    print("load X")
    X = load_matrix("tempdata/matrixX")
    print("load D")
    D = load_matrix("tempdata/matrixD")
    I = eye(X.shape[0], format="csc")
    ID = I+D
    print("start svd")
    u = sparsesvd(X,ID, VECTOR_LENGTH)[0].T
    print("finish svd")
    print("output vectors")
    output = open("tempdata/en_result.txt", "w",encoding="utf-8")
    output.write(str(len(en_word2index))+" "+str(VECTOR_LENGTH)+"\n")
    for word in en_word2index:
        vector = u[en_word2index[word]]
        output.write(word)
        for i in range(VECTOR_LENGTH):
            output.write(" %.8f"%vector[i])
        output.write("\n")
    output.close()

    output = open("tempdata/de_result.txt", "w",encoding="utf-8")
    output.write(str(len(de_word2index))+" "+str(VECTOR_LENGTH)+"\n")
    for word in de_word2index:
        vector = u[de_word2index[word]]
        output.write(word)
        for i in range(VECTOR_LENGTH):
            output.write(" %.8f"%vector[i])
        output.write("\n")
    output.close()
if __name__ =="__main__":
    svd()