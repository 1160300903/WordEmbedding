from corpus2vocab import read_vocab
if __name__ == "__main__":
    vocab_path = "tempdata/vocab/F10-W5.1en"
    input_file = open("../monolingual/vector/F10-W5.1en", "r", encoding="utf-8")
    output_file = open("../monolingual/vector/F10-W5.1en-40k", "w", encoding="utf-8")
    word2index = read_vocab(vocab_path)
    output_file.write("40000 " + input_file.readline().split()[1]+"\n")
    i = 0
    while i < 40000:
        line = input_file.readline()
        word = line.split()[0]
        if word in word2index:
            output_file.write(line)
            i += 1
    output_file.close()
    input_file.close()
