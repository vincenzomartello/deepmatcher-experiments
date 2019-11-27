import math
import torch.nn.functional as F
from torch import unsqueeze


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


def nearest_neighbor_onAttribute(v,batch_list,attribute_idx,attribute_lenght,distance_type):
    distances = []
    start_index = attribute_idx*attribute_lenght
    end_index = start_index+attribute_lenght
    for batch in batch_list:
        for sample in batch:
            if distance_type == 'cosine':
                distances.append(1-F.cosine_similarity(v[start_index:end_index]
                                                     ,sample[start_index:end_index],dim=0).data[0])
            elif distance_type == 'euclidean':
                distances.append(F.pairwise_distance(unsqueeze(v[start_index:end_index],0),
                                                     unsqueeze(sample[start_index:end_index],0)).data[0][0])
    
    best = min(distances)
    return distances.index(best)