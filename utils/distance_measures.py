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


def nearestNeighborOnAttributes(dataset,perturbations,opposite_label_data,attributes,
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


def closestDistanceOnAttribute(v,oppositeData,attribute_idx,attribute_length,distance_type='cosine'):
    start_index = attribute_idx*attribute_length
    end_index = start_index+attribute_length
    if distance_type == 'cosine':
        distances = F.cosine_similarity(v[start_index:end_index],oppositeData[:,start_index:end_index],dim=-1).data.cpu().numpy()
        return max(distances)
    else:
        distances = F.pairwise_distance(torch.unsqueeze(v[start_index:end_index],0),\
                                         oppositeData[:,start_index:end_index]).data.cpu().numpy()
        return min(distances)


def smallestDistanceOnAttributes(dataset,perturbations,opposite_label_data,attributes,
                                           attribute_length,distance_type='cosine'):
    #smallest distance from the closest real point for each attribute
    smallest_distances = []
    allOpposites = list(map(lambda v:torch.unsqueeze(v,0),opposite_label_data.values()))
    oppositeSamples = torch.cat(allOpposites)
    for sampleid in tqdm(dataset.keys()):
        sample = dataset[sampleid]
        current_smallest_distances = list(map(lambda att: closestDistanceOnAttribute
                                                 (sample+perturbations[sampleid][attributes.index(att)]
                                                                ,oppositeSamples,attributes.index(att),
                                                                attribute_length,distance_type),attributes))
        smallest_distances.append(current_smallest_distances)
    smallest_distances_df = pd.DataFrame(data = smallest_distances, columns = attributes)
    smallest_distances_df['sample_id'] = list(dataset.keys())
    return smallest_distances_df


def correctRankings(ri_norms,nn_distances):
    ri_norms = ri_norms.sort_values(by=['sample_id'])
    nn_distances = nn_distances.sort_values(by=['sample_id'])
    corrected_rankings = pd.DataFrame(ri_norms.values*nn_distances.values, 
                                        columns= ri_norms.columns, index= ri_norms.index)
    return corrected_rankings