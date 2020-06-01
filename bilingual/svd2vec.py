from corpus2vocab import read_vocab
import setting as st
import numpy as np
import sys


def svd2vec(src_vocab, trg_vocab, svd_path, src_vec, trg_vec):
    src_word2index = read_vocab(src_vocab)
    trg_word2index = read_vocab(trg_vocab)

    u = np.loadtxt(svd_path + "-U", dtype="float32")
    s = np.loadtxt(svd_path+"-s", dtype= "float32")
    # v = np.loadtxt(svd_path+"-V", dtype= "float32")

    for i in range(len(s)):
        u[:, i] *= np.sqrt(s[i])

    top = 40000
    print("output vectors")
    length = min(st.VECTOR_LENGTH, u.shape[1])
    output = open(src_vec, "w", encoding="utf-8")
    output.write(str(top) + " " + str(length) + "\n")
    for word in src_word2index:
        if src_word2index[word] >= top:
            continue
        vector = u[src_word2index[word]]
        output.write(word)
        for i in range(length):
            output.write(" %.8f" % vector[i])
        output.write("\n")
    output.close()

    total_top = top + len(src_word2index)
    output = open(trg_vec, "w", encoding="utf-8")
    output.write(str(top) + " " + str(length) + "\n")
    for word in trg_word2index:
        if trg_word2index[word] >= total_top:
            continue
        vector = u[trg_word2index[word]]
        output.write(word)
        for i in range(length):
            output.write(" %.8f" % vector[i])
        output.write("\n")
    output.close()


if __name__ == "__main__":
    src_vocab = st.VOCAB_DIR + sys.argv[1] if len(sys.argv) > 1 else st.VOCAB_DIR + "F10-W5.2en"
    trg_vocab = st.VOCAB_DIR + sys.argv[2] if len(sys.argv) > 2 else st.VOCAB_DIR + "F10-W5.2zh"

    svd_path = st.RES_DIR + sys.argv[3] if len(sys.argv) > 3 else st.RES_DIR + "F10-W5-T130-L300.en-zh-csls"

    para = svd_path.split("/")[-1].split(".")[0]
    src_vec = st.VEC_DIR + para + "." + sys.argv[4] if len(sys.argv) > 4 else st.VEC_DIR + para + "." + "2en-csls"
    trg_vec = st.VEC_DIR + para + "." + sys.argv[5] if len(sys.argv) > 5 else st.VEC_DIR + para + "." + "2zh-csls"

    svd2vec(src_vocab, trg_vocab, svd_path, src_vec, trg_vec)
