import numpy as np
import re
import nltk
from math import log2
from nltk.corpus import stopwords
from scipy.sparse import dok_matrix, csc_matrix
from counts2vocab import read_vocab
VECTOR_LENGTH = 300
UPDATE_THRESHOLD = 100000
INPUT_FILE1 = "en-counts.txt"
INPUT_FILE2 = "de-counts.txt"

def form_matrix(text, embedding_words, context_words, counts, embedding_bias, context_bias):
    tmp_counts = dok_matrix(counts.shape,dtype="float32")
    times = 0
    for k in range(len(text)):
        word, context, count = text[k]
        tmp_counts[embedding_words[word],context_words[context]] = int(count)
        times += 1
        if times == UPDATE_THRESHOLD:
            counts = counts + tmp_counts.tocsc()
            tmp_counts = dok_matrix(counts.shape,dtype="float32")
            times = 0
    counts = counts + tmp_counts.tocsc()
def compute_X():
    en_file = open(INPUT_FILE1,"r",encoding="UTF-8-sig")
    en_text = en_file.readlines()
    en_file.close()
    de_file = open(INPUT_FILE2,"r",encoding="UTF-8-sig")
    de_text = de_file.readlines()
    de_file.close()

    en_embedding = read_vocab("en_embedding.txt")
    en_context = read_vocab("en_context.txt")
    de_embedding = read_vocab("de_embedding.txt")
    de_context = read_vocab("de_context.txt")
    embedding_count = len(en_embedding) + len(de_embedding)
    context_count = len(en_context) + len(de_context)
    counts = csc_matrix((embedding_count,context_count),dtype="float32")
    form_matrix(en_text, en_embedding, en_context, counts, 0, 0)
    form_matrix(de_text, de_embedding, de_context, counts, len(en_embedding), len(en_context))

    #calculate e^pmi
    sum_r = np.array(counts.sum(axis=1))[:,0]
    sum_c = np.array(counts.sum(axis=0))[0,:]

    sum_total = sum_c.sum()
    sum_r = np.reciprocal(sum_r)
    sum_c = np.reciprocal(sum_c)

    pmi = csc_matrix(counts)
    
    normalizer = dok_matrix((len(sum_r),len(sum_r)))
    normalizer.setdiag(sum_r)
    pmi = normalizer.tocsc().dot(pmi)

    normalizer = dok_matrix((len(sum_c),len(sum_c)))
    normalizer.setdiag(sum_c)
    pmi = pmi.dot(normalizer.tocsc())

    pmi = pmi * sum_total
    pmi.data = np.log(pmi.data)

    return pmi
