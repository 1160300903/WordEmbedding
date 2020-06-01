from corpus2vocab import read_vocab
if __name__ == "__main__":
    input_file = open("vector/F10-W5.2zh-PPMI", "r", encoding="utf-8")
    output_file = open("vector/F10-W5.2zh-PPMI-40k", "w", encoding="utf-8")
    output_file.write("40000 " + input_file.readline().split()[1]+"\n")
    i = 0
    while i < 40000:
        line = input_file.readline()
        output_file.write(line)
        i += 1
    output_file.close()
    input_file.close()
