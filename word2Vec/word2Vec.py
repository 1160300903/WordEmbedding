
from gensim.models import word2vec
sentences = word2vec.LineSentence("test.txt")
model  = word2vec.Word2Vec(sentences,size = 300,min_count =5,window =5) 
model.save("word2Vec.model")
model.wv.save_word2vec_format('word2Vec.model.txt', binary=False)
