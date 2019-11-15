INPUT_FILE1 = "en-counts.txt"
INPUT_FILE2 = "de-counts.txt"
def assign_index(text, word2index, bias = 0):
    word_count = 0
    for k in range(len(text)):
        text[k] = text[k].strip().split()
        if text[k][0] not in word2index:
            word2index[text[k][0]] = word_count+bias
            word_count += 1
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
    en_file = open(INPUT_FILE1,"r",encoding="UTF-8-sig")
    en_text = en_file.readlines()
    en_file.close()
    en_word2index = {}
    assign_index(en_text, en_word2index)
    write_vocab(en_word2index,"en_word2index.txt")

    de_file = open(INPUT_FILE2,"r",encoding="UTF-8-sig")
    de_text = de_file.readlines()
    de_file.close()
    de_word2index = {}
    assign_index(de_text, de_word2index, bias = len(en_word2index))
    write_vocab(de_word2index, "de_word2index.txt")
    print("english words", len(en_word2index), len(en_word2index))
    print("german words", len(de_word2index), len(de_word2index))

if __name__ =="__main__":
    counts2vocab()