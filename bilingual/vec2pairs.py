import numpy as np
import numpy.linalg as LA
import gensim
from counts2vocab import read_vocab
from scipy.sparse import dok_matrix
EN_VEC_PATH = "mapped-model-en.txt"
DE_VEC_PATH = "mapped-model-de.txt"
TOP_TRANS = 10
def random_vector():
    en_word2index = read_vocab("en_word2index.txt")
    de_word2index = read_vocab("de_word2index.txt")
    with open("mapped-model-en.txt","w",encoding="utf-8") as output:
        output.write(str(len(en_word2index))+" "+str(300)+"\n")
        for word in en_word2index:
            output.write(word)
            vector = np.random.random(300)
            for i in range(300):
                output.write(" "+str(vector[i]))
            output.write("\n")
    with open("mapped-model-de.txt","w",encoding="utf-8") as output:
        output.write(str(len(de_word2index))+" "+str(300)+"\n")
        for word in de_word2index:
            output.write(word)
            vector = np.random.random(300)
            for i in range(300):
                output.write(" "+str(vector[i]))
            output.write("\n")

#since the embedding words set equals the context words set in my experiment,
#I just use embedding words to form D_1 and D_2
def vec2pairs():
    print("loading english vectors")
    en_vectors = gensim.models.KeyedVectors.load_word2vec_format(EN_VEC_PATH, binary = False)
    print("loading german vectors")
    de_vectors = gensim.models.KeyedVectors.load_word2vec_format(DE_VEC_PATH, binary = False)
    print("load en_word2index")
    en_word2index = read_vocab("en_word2index.txt")
    print("load de_word2index")
    de_word2index = read_vocab("de_word2index.txt")
    length = len(en_word2index)
    de_word2index = {word:de_word2index[word]-length for word in de_word2index}
    en_index2word = {en_word2index[word]:word for word in en_word2index}
    de_index2word = {de_word2index[word]:word for word in de_word2index}

    de_matrix = np.zeros((de_vectors.vector_size,len(de_word2index)))
    en_matrix = np.zeros((len(en_word2index),en_vectors.vector_size))

    print("form english matrix")
    for word in de_word2index:
        de_matrix[:,de_word2index[word]] = de_vectors[word]

    de_vec_length = np.reciprocal(LA.norm(de_matrix, axis=0))
    for i in range(len(de_vec_length)):
        de_matrix[:, i] *= de_vec_length[i]

    print("form german matrix")
    for word in en_word2index:
        en_matrix[en_word2index[word],:] = en_vectors[word]
    
    en_vec_length = np.reciprocal(LA.norm(en_matrix, axis=1))
    for i in range(len(en_vec_length)):
        en_matrix[i, :] *= en_vec_length[i]


    batch = 1000
    output = open("pairs.txt","w",encoding="utf-8")
    for i in range(int(len(en_word2index)/batch)):
        print("batch",i)
        temp = en_matrix[i*batch:i*batch+batch,:]
        result = np.dot(temp, de_matrix)
        index = np.argpartition(result,-TOP_TRANS,axis=1)[:,-TOP_TRANS:]
        base = i*batch
        for j in range(0,batch):
            sum = 0
            for k in range(TOP_TRANS):
                sum += result[j, index[j, k]]
            for k in range(TOP_TRANS):
                output.write(en_index2word[base+j]+" "+de_index2word[index[j, k]]+" "+str(result[j,index[j, k]]/sum)+"\n")

    i = int(len(en_word2index)/batch)
    temp = en_matrix[i*batch:len(en_word2index), :]
    result = np.dot(temp, de_matrix)
    index = np.argpartition(result, -TOP_TRANS, axis=1)[:,-TOP_TRANS:]
    base = i*batch
    for j in range(0, temp.shape[0]):
        sum = 0
        for k in range(TOP_TRANS):
            sum += result[j, index[j, k]]
        for k in range(TOP_TRANS):
            output.write(en_index2word[base+j]+" "+de_index2word[index[j, k]]+" "+str(result[j, index[j, k]]/sum)+"\n")
    output.close()
if __name__ == "__main__":
    vec2pairs()

    