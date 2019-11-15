import numpy as np 
from counts2vocab import read_vocab
from scipy.sparse import dok_matrix, csc_matrix
UPDATE_THRESHOLD = 5000
def form_matrix(text, en_word2index, de_word2index, counts):
    tmp_counts = dok_matrix(counts.shape,dtype="float32")
    times = 0
    for k in range(len(text)):
        en_word, de_word, count = text[k]
        tmp_counts[en_word2index[en_word], de_word2index[de_word]] = int(count)
        times += 1
        if times == UPDATE_THRESHOLD:
            counts = counts + tmp_counts.tocsc()
            tmp_counts = dok_matrix(counts.shape,dtype="float32")
            times = 0
    counts = counts + tmp_counts.tocsc()
def compute_D():
    en_word2index = read_vocab("en_embedding.txt")
    de_word2index = read_vocab("de_embedding.txt")
    length = len(en_word2index)+len(de_word2index)
    D = csc_matrix((length, length),dtype="float32")
    input = open("pairs.txt", "r",encoding="UTF-8-sig")
    pairs = [[line.strip().split()] for line in input.readlines()]
    form_matrix(pairs, en_word2index, de_word2index, D)
    return D

    