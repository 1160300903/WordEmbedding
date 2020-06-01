from corpus2vocab import read_vocab
from sparsesvd import sparsesvd
from scipy.sparse import eye
from matrix_sl import load_matrix
import setting as st
import sys
import numpy as np
def svd(matrixX, matrixD, svd_output):

    print("load X")
    X = load_matrix(matrixX)
    print("load D")
    D = load_matrix(matrixD)
    I = eye(X.shape[0], format="csc")
    print(D.shape)
    print(I.shape)
    ID = I+D
    print("start svd")
    ut, s, vt = sparsesvd(X,ID, st.VECTOR_LENGTH)
    print("finish svd")

    np.savetxt(svd_output+"-U", ut.T, fmt = "%.9f")
    np.savetxt(svd_output+"-s", s, fmt = "%.9f")
    np.savetxt(svd_output+"-V", vt.T, fmt = "%.9f")


if __name__ =="__main__":
    matrixX = st.MATRIX_DIR + sys.argv[1] if len(sys.argv) > 1 else st.MATRIX_DIR + "F10-W5.X-en-zh"
    matrixD = st.MATRIX_DIR + sys.argv[2] if len(sys.argv) > 2 else st.MATRIX_DIR + "mapped-T90.D-en-zh"
    
    para = matrixX.split("/")[-1].split(".")[0]
    para +="-" +  matrixD.split("/")[-1].split(".")[0].split("-")[-1]
    svd_output = st.RES_DIR + para + "-L" + str(st.VECTOR_LENGTH) + "."
    svd_output += sys.argv[3] if len(sys.argv) > 3 else "en-zh"

    svd(matrixX, matrixD, svd_output)
