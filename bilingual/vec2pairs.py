import numpy as np
import gensim
from counts2vocab import read_vocab
EN_VEC_PATH = "mapped-model-en.txt"
DE_VEC_PATH = "mapped-model-de.txt"
#since the embedding words set equals the context words set in my experiment,
#I just use embedding words to form D_1 and D_2
def vec2pairs():
    en_vectors = gensim.models.KeyedVectors.load_word2vec_format(EN_VEC_PATH, binary = False)
    de_vectors = gensim.models.KeyedVectors.load_word2vec_format(EN_VEC_PATH, binary = False)
    en_word2index = read_vocab("en_embedding.txt")
    de_word2index = read_vocab("de_embedding.txt")
    en_index2word = {en_word2index[word]:word for word in en_word2index}
    de_index2word = {de_word2index[word]:word for word in de_word2index}

    de_matrix = np.zeros((de_vectors.vector_size,len(de_word2index)))
    en_matrix = np.zeros((len(en_word2index),en_vectors.vector_size))
    for word in de_word2index:
        de_matrix[:,de_word2index[word]] = de_vectors[word]
    for word in en_word2index:
        en_matrix[en_word2index[word],:] = en_vectors[word]

    batch = 1000
    output = open("pairs.txt","w")
    for i in range(int(len(en_word2index)/batch)):
        temp = en_matrix[i*batch:i*batch+batch,:]
        result = np.dot(temp, de_matrix)
        index = np.argpartition(result,-10,axis=1)[:,-10:]
        base = i*batch
        for j in range(0,batch):
            sum = 0
            for k in range(10):
                sum += result[j, index[j, k]]
            for k in range(10):
                output.write(en_index2word[j+base]+" "+de_index2word[index[j, k]]+" "+str(result[j,index[j, k]]/sum)+"\n")
    
    i = int(len(en_word2index)/batch)
    temp = en_matrix[i*batch,len(en_word2index), :]
    result = np.dot(temp, de_matrix)
    index = np.argpartition(result, -10, axis=1)[:,-10:]
    base = i*batch
    for j in range(0, temp.shape[0]):
        sum = 0
        for k in range(10):
            sum += result[j, index[j, k]]
        for k in range(10):
            output.write(en_word2index[base+j]+" "+de_word2index[index[j, k]]+" "+str(result[j, index[j, k]/sum)+"\n")

    

    