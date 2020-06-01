import numpy as np
import setting as st
from scipy.sparse import dok_matrix, csc_matrix, eye
from sparsesvd import sparsesvd
import sys
from corpus2vocab import read_vocab
import time

def cnt2svd(count_file, vocab_file, PPMI):
    with open(count_file,"r",encoding="UTF-8-sig") as src_file:
        text = src_file.readlines()

    word2index = read_vocab(vocab_file)

    print("length of word_dict: "+str(len(word2index)))

    
    counts = csc_matrix((len(word2index),len(word2index)),dtype="float32")
    tmp_counts = dok_matrix((len(word2index),len(word2index)),dtype="float32")
    times = 0
    for i in range(len(text)):
        word, context, count = text[i].strip().split()
        tmp_counts[word2index[word], word2index[context]] = int(count)
        times += 1
        if times == st.UPDATE_THRESHOLD:
            counts = counts + tmp_counts.tocsc()
            tmp_counts = dok_matrix((len(word2index),len(word2index)),dtype="float32")
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

    if PPMI:
        pmi[pmi < 0] = 0
    
    I = eye(pmi.shape[0], format="csc")
    print("start svd")
    start = time.time()
    ut, s = sparsesvd(pmi, I, st.VECTOR_LENGTH)[:2]

    if PPMI:
        for i in range(len(s)):
            ut[i, :] *= np.sqrt(s[i])
    else:
        for i in range(len(s)):
            ut[i, :] *= s[i]

    print(time.time() - start)
    return ut.T, word2index

if __name__ == "__main__":
    count_file = st.CNT_DIR + sys.argv[1] if len(sys.argv) > 1 else st.CNT_DIR + "F10-W5.2zh"
    vocab_file = st.VOCAB_DIR + sys.argv[2] if len(sys.argv) > 2 else st.VOCAB_DIR + "F10-W5.2zh"

    param = count_file.split("/")[-1].split(".")[0]
    output_file = st.VEC_DIR + param + "."
    output_file += sys.argv[3] if len(sys.argv) > 3 else "2zh-PPMI"

    PPMI = bool(sys.argv[4]) if len(sys.argv) > 4 else True

    u, word2index = cnt2svd(count_file, vocab_file, PPMI)


    output = open(output_file,"w",encoding="utf-8")
    output.write(str(len(word2index)) + " " + str(st.VECTOR_LENGTH) + "\n")
    for word in word2index:
        array = u[word2index[word]]
        output.write(word)
        for i in range(st.VECTOR_LENGTH):
            output.write(" %.8f"%array[i])
        output.write("\n")
    output.close()
