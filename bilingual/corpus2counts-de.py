import numpy as np
import nltk
WORD_FREQ = 20
WINDOW_LENGTH = 5
INPUT_FILE = "../../data/mono.tok.de"
OUTPUT_FILE = "tempdata/de-counts.txt"
def main_func():
    de_file = open(INPUT_FILE,"r",encoding="UTF-8-sig")
    text = de_file.readlines()
    de_file.close()
    all_word_dict = {}
    for k in range(len(text)):
        if k%100000==0:
            print(k)
        text[k] = nltk.word_tokenize(text[k])
        for word in text[k]:
            if not word.isalpha():
                continue
            if word in all_word_dict:
                all_word_dict[word]+=1
            else:
                all_word_dict[word]=1
    vocabulary = set()
    for word in all_word_dict:
        freq = all_word_dict[word]
        if freq >= WORD_FREQ:
            vocabulary.add(word)
    
    print("length of text: "+str(len(text)))
    print("length of vocabulary: "+str(len(vocabulary)))
    counts = {word:{} for word in vocabulary}
    for i in range(len(text)):
        line = text[i]
        length  = len(line)
        if i%100000==0:
            print("count",i)
        for j in range(length):
            for k in range(1,WINDOW_LENGTH+1):
                if j - k >= 0 and line[j] in vocabulary and line[j - k] in vocabulary:
                    if line[j - k] in counts[line[j]]:
                        counts[line[j]][line[j - k]] += 1
                    else:
                        counts[line[j]][line[j - k]] = 1
                if j + k < length and line[j] in vocabulary and line[j + k] in vocabulary:
                    if line[j + k] in counts[line[j]]:
                        counts[line[j]][line[j + k]] += 1
                    else:
                        counts[line[j]][line[j + k]] = 1
                
    with open(OUTPUT_FILE,"w", encoding="utf-8") as output:
        for word in counts:
            for context in counts[word]:
                output.write(word+" "+context+" "+str(counts[word][context])+"\n")
    
    output.close()
main_func()

