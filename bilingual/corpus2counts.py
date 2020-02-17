import nltk
import setting as st


def cps2cnt(src_dir, src_file):
    word_dict = {}

    with open(src_dir + src_file, "r", encoding="UTF-8-sig") as src_file:
        text = src_file.readlines()
        for k in range(len(text)):
            if k % 10000 == 0:
                print(k, end=" ")
            text[k] = nltk.word_tokenize(text[k])
            for word in text[k]:
                if not word.isalpha():
                    continue
                word_dict[word] = 1 if word not in word_dict else word_dict[word] + 1

    vocab = set([word for word in word_dict if word_dict[word] >= st.WORD_FREQ])

    print("length of text: " + str(len(text)))
    print("length of vocabulary: " + str(len(vocab)))
    counts = {word: {} for word in vocab}

    for i in range(len(text)):
        line = text[i]
        length = len(line)
        if i % 10000 == 0:
            print("count", i, end=" ")
        for j in range(length):
            for k in range(1, st.WINDOW_LENGTH + 1):
                if j - k >= 0 and line[j] in vocab and line[j - k] in vocab:
                    counts[line[j]][line[j - k]] = counts[line[j]][line[j - k]] + 1 if line[j - k] \
                                                                                       in counts[line[j]] else 1

                if j + k < length and line[j] in vocab and line[j + k] in vocab:
                    counts[line[j]][line[j + k]] = counts[line[j]][line[j + k]] + 1 if line[j + k] \
                                                                                       in counts[line[j]] else 1

    with open(st.CNT_OUTPUT + "F" + str(st.WORD_FREQ) + "-W" + str(st.WINDOW_LENGTH) + "." + st.SRC_FILE,
              "w") as output:
        for word in counts:
            for context in counts[word]:
                output.write(word + " " + context + " " + str(counts[word][context]) + "\n")


if __name__ == "__main__":
    cps2cnt(st.SRC_DIR, st.SRC_FILE)
