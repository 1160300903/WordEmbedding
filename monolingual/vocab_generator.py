import random
input1 = open("vector/F10-W5.2en-PPMI-40k", 'r')
input2 = open("vector/F10-W5.2zh-PPMI-40k", "r")
input3 = open("../../data/dictionaries/en-zh.txt", 'r')
output = open("../../data/dictionaries/en-zh.train-25.txt", 'w')
vocab1, vocab2 = set(), set()
line1, line2 = input1.readline(), input2.readline()
while True:
    line1 = input1.readline()
    if line1 == "":
        break
    vocab1.add(line1.split(maxsplit=1)[0])

while line2 != "":
    line2 = input2.readline()
    if line2 == "":
        break
    vocab2.add(line2.split(maxsplit=1)[0])
both = []
cnt = 0
line3 = input3.readline()
while line3 != "":
    word1, word2 = line3.strip().split()
    if word1 in vocab1 and word2 in vocab2:
        both.append((word1, word2))
    line3 = input3.readline()
random.shuffle(both)
for i in range(25):
    output.write(both[i][0] + " " + both[i][1] + "\n")
