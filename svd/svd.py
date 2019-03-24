import numpy as np
import re
import nltk
from math import log2
TOP_WORD_SIZE = 100
VECTOR_LENGTH = 300
WORD_FREQ = 5
CONTEXT_FREQ = 10
def removeTopFreqWord(wordDict):
    a = list(sorted(wordDict.items(), key=lambda d:d[1], reverse = True))
    length = min(len(a),TOP_WORD_SIZE)
    stopWords = set()
    for i in range(length):
        stopWords.add(a[i][0])
    return stopWords
def getMatrix():
    enFile = open("test.txt","r",encoding="UTF-8-sig")
    text = enFile.readlines()
    enFile.close()
    wordDict = {}
    for k in range(len(text)):
        words = nltk.word_tokenize(text[k])
        for word in words:
            if word in wordDict:
                wordDict[word]+=1
            else:
                wordDict[word]=1
    stopWords = removeTopFreqWord(wordDict)
    contextDist = {}
    wordDist = {}
    i = 0
    j = 0
    for word in wordDict:
        freq = wordDict[word]
        if freq > WORD_FREQ and word.isalpha():
            wordDist[word] = i
            i+=1
        if freq > CONTEXT_FREQ and word not in stopWords:
            contextDist[word] = j
            j+=1
    del wordDict
    print("length of text: "+str(len(text)))
    print("length of wordDist: "+str(len(wordDist)))
    print("length of contextDist: "+str(len(contextDist)))
    x = np.zeros((len(wordDist),len(contextDist)),dtype="float32")
    for line in text:
        words = nltk.word_tokenize(line)
        length  = len(words)
        for j in range(length):
            if j!=0 and words[j] in wordDist and words[j-1] in contextDist:
                x[wordDist[words[j]]][contextDist[words[j-1]]]+=1
            if j!=length-1 and words[j] in wordDist and words[j+1] in contextDist:
                x[wordDist[words[j]]][contextDist[words[j+1]]]+=1
    contextLength = len(contextDist)
    del contextDist
    del text
    x += 1e-8
    total = x.sum()
    wordSum = x.sum(axis=1)
    contextSum = x.sum(axis=0)
    for r in range(len(wordDist)):
        for c in range(contextLength):
            x[r][c] = log2(x[r][c])
    x += log2(total)
    for r in range(len(wordDist)):
        x[r,:] -= log2(wordSum[r])
    for c in range(contextLength):
        x[:,c] -= log2(contextSum[c])
    return (x,wordDist)
def getReducedDimension(x,wordDist):
    la = np.linalg
    U, X = la.svd(x,full_matrices=0)[:2]
    length = VECTOR_LENGTH if X.shape[0]>VECTOR_LENGTH else X.shape[0]
    U = U[:,:length]
    print(X.shape)
    X = X[:length]
    for i in range(length):
        U[:,i]*= X[i]
    return U
x, wordDist = getMatrix()
U = getReducedDimension(x, wordDist)
output = open("result1.txt","w",encoding="utf-8")
output.write(str(len(wordDist))+" "+str(VECTOR_LENGTH)+"\n")
for word in wordDist:
    array = U[wordDist[word]]
    output.write(word)
    for i in range(VECTOR_LENGTH):
        output.write(" %.8f"%array[i])
    output.write("\n")
output.close()
