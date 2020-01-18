import math
import torch.nn.functional as F
from torch import unsqueeze
import torch
import numpy as np
from tqdm import tqdm
import pandas as pd


def euclidean_distance_with_max_difference_dimension(v,q):
    distance = 0
    max_difference = -1
    max_difference_dimension = 0
    for idx,vi in enumerate(v):
        qi = q[idx]
        diff = qi-vi
        if(diff>max_difference):
            max_difference = diff
            max_difference_dimension = idx
        distance += (diff)**2
    return math.sqrt(distance),max_difference_dimension


#for each positive sample calculate closer negative sample and its index
def calculate_closer_vector(pos_vector_list,neg_vector_list):
    #mi salvo l'indice del vettore più vicino come chiave
    closer_vectors = []
    for curr_pos_batch in pos_vector_list:
        for curr_positive in curr_pos_batch:
            print('proccessing vector')
            current_min = 100000000
            index = 1
            closer_index = -1
            max_dim = 0
            for batch in neg_vector_list:
                for curr_negative in batch:
                    curr_distance,dim_max = euclidean_distance_with_max_difference_dimension
                    (curr_positive.data,curr_negative.data)
                    if(curr_distance<current_min):
                        current_min=curr_distance
                        closer_index = index
                        max_dim = dim_max
                    index +=1
            closer_vectors.append((closer_index, current_min,dim_max))
    return closer_vectors


def nearest_neighbor(v,batch_list,distance_type):
    distances = []
    for batch in batch_list:
        for sample in batch:
            if distance_type=='euclidean':
                distances.append(F.pairwise_distance(v,sample))
            elif distance_type == 'cosine':
                distances.append(F.cosine_similarity(v,sample))
    closer = min(distances)
    return distances.index(closer)


def nearestNeighborOnAttribute(v,oppositeData,attribute_idx,attribute_length,min_similarity):
    start_index = attribute_idx*attribute_length
    end_index = start_index+attribute_length
    distances = F.cosine_similarity(v[start_index:end_index],oppositeData[:,start_index:end_index],dim=-1).data.cpu().numpy()
    best = max(distances)
    ## to exclude neighbors too far we return not existing id
    if best< min_similarity:
        return -1
    else:
        ## we only get the first value
        return (np.where(distances == best)[0][0])


def convertToIds(index,id_list):
    if index == -1:
        return index
    else:
        return id_list[index]


def nearestNeighborsOnAttribute(dataset,perturbations,opposite_label_data,attributes,
                                           attribute_length,min_similarity=-1):
    #lista di tuple: vettore più vicino considerando tutti gli elementi e closer solo secondo un attributo
    closer_vectors = []
    allOpposites = list(map(lambda v:torch.unsqueeze(v,0),opposite_label_data.values()))
    oppositeSamples = torch.cat(allOpposites)
    for sampleid in tqdm(dataset.keys()):
        sample = dataset[sampleid]
        current_closer_vectors = list(map(lambda att: nearestNeighborOnAttribute
                                                 (sample+perturbations[sampleid][attributes.index(att)]
                                                                ,oppositeSamples,attributes.index(att),
                                                                attribute_length,min_similarity),attributes))
        closer_vectors.append(current_closer_vectors)
    closer_vectors_df = pd.DataFrame(data = closer_vectors, columns = attributes)
    opposite_ids = list(opposite_label_data.keys())
    closer_vectors_df = closer_vectors_df.applymap(lambda c:convertToIds(c,opposite_ids))
    closer_vectors_df['SampleID'] = list(dataset.keys())
    return closer_vectors_df

