from counts2matrixX import compute_X
from pairs2matrixD import compute_D
from counts2vocab import read_vocab
import numpy as np
from sparsesvd import sparsesvd
from scipy.sparse import csc_matrix, eye
VECTOR_LENGTH = 300
def svd():
    print("load word2index")
    en_word2index = read_vocab("en_word2index.txt")
    de_word2index = read_vocab("de_word2index.txt")
    print("compute X")
    X = compute_X()
    print("compute D")
    D = compute_D()
    I = eye(X.shape[0], format="csc")
    ID = I+D
    print("multipy D+I and X")
    X = np.dot(ID, X)
    X = np.dot(X, ID.T)
    X *= 1/4
    print("start svd")
    u = sparsesvd(X, VECTOR_LENGTH)[0].T
    print("finish svd")
    print("output vectors")
    output = open("en_result.txt", "w",encoding="utf-8")
    output.write(str(len(en_word2index))+" "+str(VECTOR_LENGTH)+"\n")
    for word in en_word2index:
        vector = u[en_word2index[word]]
        output.write(word)
        for i in range(VECTOR_LENGTH):
            output.write(" %.8f"%vector[i])
        output.write("\n")
    output.close()

    output = open("de_result.txt", "w",encoding="utf-8")
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