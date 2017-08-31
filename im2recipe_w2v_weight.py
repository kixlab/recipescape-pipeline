import word2vec
import numpy as np

modelfile = 'vocab.bin'

model = word2vec.load(modelfile)

#model.vocab
#model.vectors.shape
#model.vectors

# word1='add'
# word2='Add'

def cosine_similarity(word1, word2):
    try:
        return np.dot(model[word1], model[word2])/(np.linalg.norm(model[word1])* np.linalg.norm(model[word2]))
    except KeyError:
        return 0
    
    #print(cosine_similarity)

