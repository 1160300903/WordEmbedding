import numpy as np
import re
import nltk
from math import log2
from nltk.corpus import stopwords
from scipy.sparse import dok_matrix, csc_matrix
from sparsesvd import sparsesvd
VECTOR_LENGTH = 300
UPDATE_THRESHOLD = 100000
INPUT_FILE = "counts.txt"
def main_func():
    en_file = open(INPUT_FILE,"r",encoding="UTF-8-sig")
    text = en_file.readlines()
    en_file.close()
    context_words = {}
    embedding_words = {}
    i = 0
    j = 0
    for k in range(len(text)):
        text[k] = text[k].strip().split()
        if text[k][0] not in embedding_words:
            embedding_words[text[k][0]] = i
            i += 1
        if text[k][1] not in context_words:
            context_words[text[k][1]] = j
            j += 1
    print("length of wordDist: "+str(len(embedding_words)))
    print("length of contextDist: "+str(len(context_words)))

    
    counts = csc_matrix((len(embedding_words),len(context_words)),dtype="float32")
    tmp_counts = dok_matrix((len(embedding_words),len(context_words)),dtype="float32")
    times = 0
    for i in range(len(text)):
        word, context, count = text[i]
        tmp_counts[embedding_words[word],context_words[context]] = int(count)
        times += 1
        if times == UPDATE_THRESHOLD:
            counts = counts + tmp_counts.tocsc()
            tmp_counts = dok_matrix((len(embedding_words),len(context_words)),dtype="float32")
            times = 0
    counts = counts + tmp_counts.tocsc()
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

    ut = sparsesvd(pmi, VECTOR_LENGTH)[0]
    return ut.T, embedding_words

u, embedding_words = main_func()
output = open("result.txt","w",encoding="utf-8")
output.write(str(len(embedding_words))+" "+str(VECTOR_LENGTH)+"\n")
for word in embedding_words:
    array = u[embedding_words[word]]
    output.write(word)
    for i in range(VECTOR_LENGTH):
        output.write(" %.8f"%array[-i-1])
    output.write("\n")
output.close()
