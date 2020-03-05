import numpy as np 
import gensim
import matplotlib.pyplot as plt
def plot_vectors(path1, word_list, d = 2, path2 = None):
    model1 = gensim.models.KeyedVectors.load_word2vec_format(path1, binary=False)
    vector_list1 = np.array([model1[word][:d] for word in word_list])
    
    if path2 is not None:
        model2 = gensim.models.KeyedVectors.load_word2vec_format(path2, binary=False)
        vector_list2 = np.array([model2[word][:d] for word in word_list])
    
    assert d == 2 or d == 3
    if d == 2:
        plt.scatter(vector_list1[:, 0], vector_list1[:, 1], c = "blue")
        if path2 is not None:
            plt.scatter(vector_list2[:, 0], vector_list2[:, 1], c = "brown")
    if d == 3:
        plt.scatter(vector_list1[:, 0], vector_list1[:, 1], vector_list1[:, 2], c = "blue")
        if path2 is not None:
            plt.scatter(vector_list2[:, 0], vector_list2[:, 1], vector_list2[:, 2], c = "brown")
    