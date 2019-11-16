import numpy as np 
from counts2vocab import read_vocab
from scipy.sparse import dok_matrix, csc_matrix
from matrix_sl import save_matrix
UPDATE_THRESHOLD = 5000
def form_matrix(text, en_word2index, de_word2index, D):
    tmp_D = dok_matrix(D.shape,dtype="float32")
    times = 0
    for k in range(len(text)):
        en_word, de_word, similarity = text[k]
        tmp_D[en_word2index[en_word], de_word2index[de_word]] = float(similarity)
        tmp_D[de_word2index[de_word], en_word2index[en_word]] = float(similarity)
        times += 1
        if times == UPDATE_THRESHOLD:
            D = D + tmp_D.tocsc()
            tmp_D = dok_matrix(D.shape,dtype="float32")
            times = 0
    D = D + tmp_D.tocsc()
    return D
def compute_D():
    en_word2index = read_vocab("tempdata/en_word2index.txt")
    de_word2index = read_vocab("tempdata/de_word2index.txt")
    length = len(en_word2index)+len(de_word2index)
    D = csc_matrix((length, length), dtype="float32")
    input = open("tempdata/pairs.txt", "r", encoding="UTF-8-sig")
    pairs = [line.strip().split() for line in input.readlines()]
    input.close()
    D = form_matrix(pairs, en_word2index, de_word2index, D)
    save_matrix("tempdata/matrixD", D)

if __name__ == "__main__":
    compute_D()

    