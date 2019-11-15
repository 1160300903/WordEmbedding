from counts2matrixX import compute_X
from pairs2matrixD import compute_D
from counts2vocab import read_vocab
import numpy as np
from sparsesvd import sparsesvd
from scipy.sparse import csc_matrix, eye
VECTOR_LENGTH = 300
def svd():
    en_words2index = read_vocab("en_embedding.txt")
    de_words2index = read_vocab("de_embedding.txt")
    X = compute_X()
    D = compute_D()
    I = eye(X.shape[0], format="csc")
    ID = I+D
    X = np.dot(ID, X)
    X = np.dot(X, ID)
    u = sparsesvd(X, VECTOR_LENGTH)[0].T

    output = open("en_result.txt", "w",encoding="utf-8")
    output.write(str(len(en_words2index))+" "+str(VECTOR_LENGTH))
    for word in en_words2index:
        vector = u[en_words2index[word]]
        output.write(word)
        for i in range(VECTOR_LENGTH):
            output.write(" %.8f"%vector[-i-1])
        output.write("\n")
    output.close()

    output = open("de_result.txt", "w",encoding="utf-8")
    output.write(str(len(de_words2index))+" "+str(VECTOR_LENGTH))
    for word in de_words2index:
        vector = u[de_words2index[word]]
        output.write(word)
        for i in range(VECTOR_LENGTH):
            output.write(" %.8f"%vector[i])
        output.write("\n")
    output.close()