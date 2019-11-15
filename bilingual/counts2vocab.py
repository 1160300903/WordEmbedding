INPUT_FILE1 = "en-counts.txt"
INPUT_FILE2 = "de-counts.txt"
def assign_index(text, embedding_words, context_words):
    embedding_count, context_count = 0, 0
    for k in range(len(text)):
        text[k] = text[k].strip().split()
        if text[k][0] not in embedding_words:
            embedding_words[text[k][0]] = embedding_count
            embedding_count += 1
        if text[k][1] not in context_words:
            context_words[text[k][1]] = context_count
            context_count += 1
def write_vocab(words,path):
    with open(path, "w") as output:
        for word in words:
            output.write(word+" "+str(words[word])+"\n")
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
    en_context_words = {}
    en_embedding_words = {}
    assign_index(en_text, en_embedding_words, en_context_words)
    write_vocab(en_embedding_words,"en_embedding.txt")
    write_vocab(en_context_words, "en_context.txt")

    de_file = open(INPUT_FILE2,"r",encoding="UTF-8-sig")
    de_text = de_file.readlines()
    de_file.close()
    de_context_words = {}
    de_embedding_words = {}
    assign_index(de_text, de_embedding_words, de_context_words)
    write_vocab(de_embedding_words, "de_embedding.txt")
    write_vocab(de_context_words,"de_context.txt")
    print("embedding words", len(en_embedding_words), len(de_embedding_words))
    print("context words", len(en_context_words), len(de_context_words))