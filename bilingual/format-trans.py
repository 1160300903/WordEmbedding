input = open("../../vecmap/data/dictionaries/en.zh.dict", 'r', encoding='utf-8')
output = open("../../vecmap/data/dictionaries/en-zh.txt", 'w', encoding='utf-8')

lines = input.readlines()
for line in lines:
    line = line.strip()
    srcs = line.split("\t")[0].strip().split(" ")
    trgs = line.split("\t")[1].strip().split(" ")
    for src in srcs:
        for trg in trgs:
            output.write(src + " " + trg + "\n")
input.close()
output.close()