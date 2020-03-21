import nltk
from collections import defaultdict
import setting as st
import sys
def assign_index(path, count_path, word2index, bias = 0):
    file = open(path,'r', encoding='utf-8')
    text = file.readlines()
    file.close()
    
    embed_words = set()
    with open(count_path, "r", encoding="utf-8") as input:
        for line in input.readlines():
            word = line.strip().split()[0]
            embed_words.add(word)

    word_counts = defaultdict(int)
    for k in range(len(text)):
        text[k] = nltk.word_tokenize(text[k])
        for word in text[k]:
            if not word.isalpha():
                continue
            word_counts[word] += 1
    word_counts = [(word, word_counts[word]) for word in embed_words]

    word_counts.sort(key = lambda a: a[1], reverse=True)
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
    with open(path,"r",encoding="UTF-8") as input:
        for line in input.readlines():
            word, index = line.strip().split()
            vocab[word] = int(index)
    return vocab
def corpus2vocab(src_file, trg_file, src_count, trg_count, output_src, output_trg):
    src_word2index = {}
    assign_index(src_file, src_count, src_word2index)
    write_vocab(src_word2index,output_src)

    trg_word2index = {}
    assign_index(trg_file, trg_count, trg_word2index)
    write_vocab(trg_word2index,output_trg)
    print("the source language words", len(src_word2index), len(src_word2index))
    print("the target language words", len(trg_word2index), len(trg_word2index))

if __name__ =="__main__":
    src_file = st.CPR_DIR + sys.argv[1] if len(sys.argv) > 1 else st.CPR_DIR + "mono.tok.en"
    trg_file = st.CPR_DIR + sys.argv[2] if len(sys.argv) > 2 else st.CPR_DIR + "mono.tok.de"

    src_count = st.CNT_DIR + sys.argv[3] if len(sys.argv) > 3 else st.CNT_DIR + "F10-W5.1en"
    trg_count = st.CNT_DIR + sys.argv[4] if len(sys.argv) > 4 else st.CNT_DIR + "F10-W5.1de"

    para = src_count.split("/")[-1].split(".")[0]
    output_src = st.VOCAB_DIR + para + "."
    output_src += sys.argv[5] if len(sys.argv) > 5 else "1en"
    
    output_trg = st.VOCAB_DIR + para + "."
    output_trg += sys.argv[6] if len(sys.argv) > 6 else "1de"

    corpus2vocab(src_file, trg_file, src_count, trg_count, output_src, output_trg)
