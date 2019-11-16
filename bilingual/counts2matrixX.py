import numpy as np
import re
import nltk
from math import log2
from nltk.corpus import stopwords
from scipy.sparse import dok_matrix, csc_matrix
from counts2vocab import read_vocab
from matrix_sl import save_matrix
VECTOR_LENGTH = 300
UPDATE_THRESHOLD = 100000
INPUT_FILE1 = "tempdata/en-counts.txt"
INPUT_FILE2 = "tempdata/de-counts.txt"

def form_matrix(text, word2index, counts):
    tmp_counts = dok_matrix(counts.shape,dtype="float32")
    times = 0
    for k in range(len(text)):
        word, context, count = text[k]
        tmp_counts[word2index[word],word2index[context]] = int(count)
        times += 1
        if times == UPDATE_THRESHOLD:
            counts = counts + tmp_counts.tocsc()
            tmp_counts = dok_matrix(counts.shape,dtype="float32")
            times = 0
    counts = counts + tmp_counts.tocsc()
    return counts
def compute_X():
    print("read counts")
    en_file = open(INPUT_FILE1,"r",encoding="UTF-8-sig")
    en_text = [line.strip().split() for line in en_file.readlines()]
    en_file.close()
    de_file = open(INPUT_FILE2,"r",encoding="UTF-8-sig")
    de_text = [line.strip().split() for line in de_file.readlines()]
    de_file.close()

    print("read word2index")
    en_word2index = read_vocab("tempdata/en_word2index.txt")
    de_word2index = read_vocab("tempdata/de_word2index.txt")
    total_number = len(en_word2index) + len(de_word2index)

    print("form matrix X")
    counts = csc_matrix((total_number,total_number),dtype="float32")
    counts = form_matrix(en_text, en_word2index, counts)
    counts = form_matrix(de_text, de_word2index, counts)
    #calculate e^pmi
    print("compute pmi")
    sum_r = np.array(counts.sum(axis=1))[:,0]
    sum_c = np.array(counts.sum(axis=0))[0,:]

    sum_total = sum_c.sum()
    sum_r = np.reciprocal(sum_r)
    sum_c = np.reciprocal(sum_c)

    pmi = csc_matrix(counts)
    print("divided by marginal sum")
    normalizer = dok_matrix((len(sum_r),len(sum_r)))
    normalizer.setdiag(sum_r)
    pmi = normalizer.tocsc().dot(pmi)

    normalizer = dok_matrix((len(sum_c),len(sum_c)))
    normalizer.setdiag(sum_c)
    pmi = pmi.dot(normalizer.tocsc())

    print("multiply total sum")
    pmi = pmi * sum_total
    pmi.data = np.log(pmi.data)

    save_matrix("tempdata/matrixX", pmi)


if __name__ == "__main__":
    compute_X()

