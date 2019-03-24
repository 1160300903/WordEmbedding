import nltk
file1 = open("mono.tok.en","r")
file2 = open("test.txt","w")
lines = file1.readlines()
lines = lines[800000:1600000]
for line in lines: 
    file2.write(line)
file1.close()
file2.close()      
