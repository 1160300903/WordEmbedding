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
    word_counts = [(word, word_counts[word]) for word in word_counts]
    word_counts.sort(lambda a: a[1], reverse=True)
    word2index = {}
    i = 0
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
def counts2vocab():
    en_word2index = {}
    assign_index(INPUT_FILE1, en_word2index)
    write_vocab(en_word2index,"tempdata/en_word2index.txt")

    de_word2index = {}
    assign_index(INPUT_FILE2, de_word2index, bias = len(en_word2index))
    write_vocab(de_word2index, "tempdata/de_word2index.txt")
    print("english words", len(en_word2index), len(en_word2index))
    print("german words", len(de_word2index), len(de_word2index))

if __name__ =="__main__":
    counts2vocab()