import numpy as np
from scipy.sparse import dok_matrix, csc_matrix
from corpus2vocab import read_vocab
from matrix_sl import save_matrix
import setting as st
import sys
def form_matrix(file, word2index, counts):
    tmp_counts = dok_matrix(counts.shape,dtype="float32")
    times = 0
    text = file.readline()
    while text != "":
        word, context, count = text.strip().split()
        tmp_counts[word2index[word],word2index[context]] = int(count)
        times += 1
        if times == st.UPDATE_THRESHOLD:
            counts = counts + tmp_counts.tocsc()
            tmp_counts = dok_matrix(counts.shape,dtype="float32")
            times = 0
        text = file.readline()
    counts = counts + tmp_counts.tocsc()
    return counts
def compute_X(src_file, trg_file, src_vocab, trg_vocab, output_file, pmi):

    print("read word2index")
    src_word2index = read_vocab(src_vocab)
    trg_word2index = read_vocab(trg_vocab)
    total_number = len(src_word2index) + len(trg_word2index)

    print("form matrix X")
    counts = csc_matrix((total_number,total_number),dtype="float32")
    counts = form_matrix(open(src_file,"r",encoding="UTF-8-sig"), src_word2index, counts)
    counts = form_matrix(open(trg_file,"r",encoding="UTF-8-sig"), trg_word2index, counts)
    if pmi:
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
        save_matrix(output_file, pmi)
    else:
        save_matrix(output_file, counts)


if __name__ == "__main__":
    src_file = st.CNT_DIR + sys.argv[1] if len(sys.argv) > 1 else st.CNT_DIR + "F10-W5.1en"
    trg_file = st.CNT_DIR + sys.argv[2] if len(sys.argv) > 2 else st.CNT_DIR + "F10-W5.1de"
    src_vocab = st.VOCAB_DIR + sys.argv[3] if len(sys.argv) > 3 else st.VOCAB_DIR + "F10-W5.1en"
    trg_vocab = st.VOCAB_DIR + sys.argv[4] if len(sys.argv) > 4 else st.VOCAB_DIR + "F10-W5.1de"
    
    para = src_file.split("/")[-1].split(".")[0]
    output_file = st.MATRIX_DIR + para + "." + sys.argv[5] if len(sys.argv) > 5 else st.MATRIX_DIR + para + "." + "X-en-de.nopmi"
    pmi = bool(sys.argv[6]) if len(sys.argv) > 6 else False
    compute_X(src_file, trg_file, src_vocab, trg_vocab, output_file, pmi = pmi)

