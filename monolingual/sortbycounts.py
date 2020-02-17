import numpy as np
import nltk
import gensim
from collections import defaultdict
VEC_PATH = ""
INPUT_FILE = "../../data/mono.tok.en.test"
VECTOR_LENGTH = 300
if __name__ == "__main__":
    en_vectors = gensim.models.KeyedVectors.load_word2vec_format(VEC_PATH, binary = False)
    en_file = open(INPUT_FILE,encoding="utf-8")
    text = en_file.readlines()
    counts = defaultdict(int)
    for i in range(len(text)):
        for word in nltk.word_tokenize(text[i]):
            counts[word] += 1
    counts_list = [(word, counts[word]) for word in counts]
    counts_list = sorted(counts_list, key = lambda a:a[1], reverse=True)
    with open("sorted_result.txt",'w',encoding="utf-8") as output:
        for word,counts in counts_list:
            output.write(word)
            array = en_vectors[word]
            for i in range(VECTOR_LENGTH):
                output.write(" %.8f"%array[i])
            output.write("\n")

