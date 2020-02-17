import numpy as np
from scipy.sparse import dok_matrix, csc_matrix
from counts2vocab import read_vocab
from matrix_sl import save_matrix
import setting as st

def form_matrix(text, word2index, counts):
    tmp_counts = dok_matrix(counts.shape,dtype="float32")
    times = 0
    for k in range(len(text)):
        word, context, count = text[k]
        tmp_counts[word2index[word],word2index[context]] = int(count)
        times += 1
        if times == st.UPDATE_THRESHOLD:
            counts = counts + tmp_counts.tocsc()
            tmp_counts = dok_matrix(counts.shape,dtype="float32")
            times = 0
    counts = counts + tmp_counts.tocsc()
    return counts
def compute_X():
    print("read counts")
    src_file = open(st.CNT_OUTPUT + st.SRC_CNT,"r",encoding="UTF-8-sig")
    src_text = [line.strip().split() for line in src_file.readlines()]
    src_file.close()
    trg_file = open(st.CNT_OUTPUT + st.TRG_CNT,"r",encoding="UTF-8-sig")
    trg_text = [line.strip().split() for line in trg_file.readlines()]
    trg_file.close()

    print("read word2index")
    en_word2index = read_vocab("tempdata/en_word2index.txt")
    de_word2index = read_vocab("tempdata/de_word2index.txt")
    total_number = len(en_word2index) + len(de_word2index)

    print("form matrix X")
    counts = csc_matrix((total_number,total_number),dtype="float32")
    counts = form_matrix(src_text, en_word2index, counts)
    counts = form_matrix(trg_text, de_word2index, counts)
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

