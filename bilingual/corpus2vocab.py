import nltk
from collections import defaultdict
import setting as st
def assign_index(path, word2index, bias = 0):
    file = open(path,'r', encoding='utf-8')
    text = file.readlines()
    file.close()
    word_counts = defaultdict(int)
    for k in range(len(text)):
        text[k] = nltk.word_tokenize(text[k])
        for word in text[k]:
            word_counts[word] += 1
    word_counts = [(word, word_counts[word]) for word in word_counts if word_counts[word] >= st.WORD_FREQ]
    # TODO DELETE word whose freq is lower than WORD_FREQ
    word_counts.sort(lambda a: a[1], reverse=True)
    i = bias
    for word, _ in word_counts:
        word2index[word] = i
        i += 1
def write_vocab(word2index, path):
    with open(path, "w", encoding="utf-8") as output:
        for word in word2index:
            output.write(word+" "+str(word2index[word])+"\n")
def read_vocab(path):
    vocab = {}
    with open(path,"r",encoding="UTF-8-sig") as input:
        for line in input.readlines():
            word, index = line.strip().split()
            vocab[word] = int(index)
    return vocab
def corpus2vocab(src_file, trg_file, output_src, output_trg):
    src_word2index = {}
    assign_index(src_file, src_word2index)
    write_vocab(src_word2index,output_src)

    trg_word2index = {}
    assign_index(trg_file, trg_word2index, bias = len(src_word2index))
    write_vocab(trg_word2index,output_trg)
    print("english words", len(src_word2index), len(src_word2index))
    print("german words", len(trg_word2index), len(trg_word2index))

if __name__ =="__main__":
    src_file, trg_file = st.CPR_DIR + "", st.CPR_DIR + ""
    output_src, output_trg = st.VOCAB_DIR + "F" + str(st.WORD_FREQ) + ".", st.VOCAB_DIR + "F" + str(st.WORD_FREQ) + "."
    corpus2vocab(src_file, trg_file, output_src, output_trg)