import numpy as np
from tqdm import tqdm
import scipy as sp


def loadGloveModel(gloveFile):
    print ("Loading Glove Model")
    f = open(gloveFile,'r')
    model = {}
    for line in f:
        row = line.strip().split(' ')
        word = row[0]
        #print(word)
        embedding = np.array([float(val) for val in row[1:]])
        model[word] = embedding
    print ("Done.",len(model)," words loaded!")
    return model


def find_closest_words(w_embedding,embedding_matrix,words,maxDistance):
    wordDistances = sp.spatial.distance.cdist(w_embedding.reshape(1,-1),embedding_matrix,'cosine')
    closestWords = []
    for i,distance in enumerate(wordDistances[0]):
        if distance <=maxDistance and distance>0:
            closestWords.append((words[i],distance))
    return closestWords


def getTopClosest(dictionary,embeddings_dict,embedding_matrix,maxDistance):
    closest = {}
    notfound = []
    words = list(embeddings_dict.keys())
    for key in tqdm(dictionary.keys()):
        if key in embeddings_dict:
            closest_words = find_closest_words(embeddings_dict[key],embedding_matrix,words,maxDistance)
            closest[key] = closest_words
        else:
            notfound.append(key)
    return closest,notfound