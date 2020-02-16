import pandas as pd
import deepmatcher as dm
from itertools import chain, combinations
import numpy as np
import random as rd
from tqdm import tqdm
from utils.deepmatcher_utils import wrapDm
from utils.sampleBuilder import buildNegativeFromSample
from strsimpy.normalized_levenshtein import NormalizedLevenshtein
from strsimpy.jaccard import Jaccard



##maxLenAttributes is the maximum number of perturbed attributes we want to consider
def aggregateRankings(ranking_l,positive,maxLenAttributes):
    allRank = {}
    for rank in ranking_l:
        for key in rank.keys():
            if len(key) <= maxLenAttributes:
                if key in allRank:
                    allRank[key] += 1
                else:
                    allRank[key] = 1
    alteredAttr = list(map(lambda t:"/".join(t),list(allRank.keys())))
    rankForHistogram = {'attributes':alteredAttr,'flipped':list(allRank.values())}
    fig_height = len(alteredAttr)
    fig_width = fig_height
    df = pd.DataFrame(rankForHistogram)
    if positive:
        ax = df.plot.barh(x='attributes', y='flipped',color='green',figsize=(fig_height,fig_width))
    else:
        ax = df.plot.barh(x='attributes', y='flipped',color='red',figsize=(fig_height,fig_width))
    return ax,allRank


def _renameColumnsWithPrefix(prefix,df):
        newcol = []
        for col in list(df):
            newcol.append(prefix+col)
        df.columns = newcol


def createOriginalTriple(r1,r2,r3):
        r1 = r1.reset_index(drop=True)
        r2 = r2.reset_index(drop=True)
        r3 = r3.reset_index(drop=True)
        r2_copy = r2.copy()
        renameColumnsWithPrefix('ltable_',r1)
        renameColumnsWithPrefix('rtable_',r2)
        renameColumnsWithPrefix('ltable_',r2_copy)
        renameColumnsWithPrefix('rtable_',r3)
        r1r2 = pd.concat([r1,r2],axis=1,sort=False)
        r2r3 = pd.concat([r2_copy,r3],axis=1,sort=False)
        r1r3 = pd.concat([r1,r3],axis=1,sort=False)
        result = pd.concat([r1r2,r2r3,r1r3])
        result['id'] = [0,1,2]
        return result

    
def _powerset(xs,minlen,maxlen):
    return [subset for i in range(minlen,maxlen+1)
            for subset in combinations(xs, i)]


def createPositiveTriangle(r1,r2,r3,attributes,maxLenAttributeSet):
        allAttributesSubsets = list(_powerset(attributes,1,maxLenAttributeSet))
        perturbations = []
        perturbedAttributes = []
        for subset in allAttributesSubsets:
            perturbedAttributes.append(subset)
            newRow = r2.copy()
            for att in subset:
                newRow[att] = r3[att]
            perturbations.append(newRow)
        perturbations_df = pd.DataFrame(perturbations)
        return perturbations_df,perturbedAttributes
    

def createNegativeTriangle(r1,r2,r3,attributes,maxLenAttributeSet):
    allAttributesSubsets = list(_powerset(attributes,1,maxLenAttributeSet))
    perturbations = []
    perturbedAttributes = []
    for subset in allAttributesSubsets:
        perturbedAttributes.append(subset)
        newRow = r3.copy()
        for att in subset:
            newRow[att] = r2[att]
        perturbations.append(newRow)
    perturbations_df = pd.DataFrame(perturbations)
    return perturbations_df,perturbedAttributes


def _occurrInNegative(sampleid,id_list,labels):
        occurrences = 0
        for i,sample_id in enumerate(id_list):
            if labels[i] == 0 and sample_id == sampleid:
                occurrences += 1
        return occurrences

    
def generateNewNegatives(df,source1,source2,newNegativesToBuild):
    allNewNegatives = []
    jaccard = Jaccard(3)
    positives = df[df.label==1]
    newNegativesPerSample = round(newNegativesToBuild/len(positives))
    for i in range(len(positives)):
        locc = _occurrInNegative(positives.iloc[i]['ltable_id'],
                                 df['ltable_id'].values,df['label'].values)
        rocc = _occurrInNegative(positives.iloc[i]['rtable_id'],df['rtable_id'].values,
                                 df['label'].values)
        if locc==0 and rocc == 0:
            permittedIds = [sampleid for sampleid in df['rtable_id'].values if sampleid!= df.iloc[i]['rtable_id']]
            newNegatives_l = buildNegativeFromSample(positives.iloc[i]['ltable_id'],permittedIds,\
                                                     newNegativesPerSample,source1,source2,jaccard,0.5)
            newNegatives_df = pd.DataFrame(data=newNegatives_l,columns=['ltable_id','rtable_id','label'])
            allNewNegatives.append(newNegatives_df)
    allNewNegatives_df = pd.concat(allNewNegatives)
    return allNewNegatives_df


def prepareDataset(dataset,source1,source2,newNegativesToBuild,left_prefix='ltable_',right_prefix='rtable_'):
    colForDrop = [col for col in list(dataset) if col not in ['id','ltable_id','rtable_id','label']]
    dataset = dataset.drop_duplicates(colForDrop)
    ##positives = dataset[dataset.label==1]
    newNegatives = generateNewNegatives(dataset,source1,source2,newNegativesToBuild)
    left_columns = []
    right_columns = []
    source1_c = source1.copy()
    source2_c = source2.copy()
    for lcol,rcol in zip(list(source1_c),list(source2_c)):
        left_columns.append(left_prefix+lcol)
        right_columns.append(right_prefix+rcol)
    source1_c.columns = left_columns
    source2_c.columns = right_columns
    pdata = pd.merge(newNegatives,source1_c, how='inner')
    newNegatives_df = pd.merge(pdata,source2_c,how='inner')
    lastDataset_id = dataset['id'].values[-1]
    newNegatives_df['id'] = np.arange(lastDataset_id+1,lastDataset_id+1+len(newNegatives_df))
    augmentedData = pd.concat([dataset,newNegatives_df])
    return augmentedData



## for now we suppose to have only two sources
def getPositiveCandidates(dataset,sources):
        triangles = []
        positives = dataset[dataset.label==1]
        negatives = dataset[dataset.label==0]
        l_pos_ids = positives.ltable_id.values
        r_pos_ids = positives.rtable_id.values
        for lid,rid in zip(l_pos_ids,r_pos_ids):         
            if _occurrInNegative(rid,dataset.rtable_id.values,dataset.label.values)>=1:
                relatedTuples = negatives[negatives.rtable_id == rid]
                for curr_lid in relatedTuples.ltable_id.values:
                    triangles.append((sources[0].iloc[lid],sources[1].iloc[rid],sources[0].iloc[curr_lid]))
            if _occurrInNegative(lid,dataset.ltable_id.values,dataset.label.values)>=1:
                relatedTuples = negatives[negatives.ltable_id==lid]
                for curr_rid in relatedTuples.rtable_id.values:
                    triangles.append((sources[1].iloc[rid],sources[0].iloc[lid],sources[1].iloc[curr_rid]))
        return triangles
    

## not used for now
def getNegativeCandidates(dataset,sources):
        triangles = []
        negatives = dataset[dataset.label==0]
        l_neg_ids = negatives.ltable_id.values
        r_neg_ids = negatives.rtable_id.values
        lr_ids = np.dstack((l_neg_ids,r_neg_ids)).squeeze()
        for lid,rid in lr_ids:
            if _occurrInNegative(rid,dataset.rtable_id.values,dataset.label.values)>=2:
                relatedTuples = negatives[negatives.rtable_id == rid]
                for curr_lid in relatedTuples.ltable_id.values:
                    if curr_lid!= lid:
                        triangles.append((sources[0].iloc[lid],sources[1].iloc[rid],sources[0].iloc[curr_lid]))
            if _occurrInNegative(lid,dataset.ltable_id.values,dataset.label.values)>=2:
                relatedTuples = negatives[negatives.ltable_id == lid]
                for curr_rid in relatedTuples.rtable_id.values:
                    if curr_rid != rid:
                        triangles.append((sources[1].iloc[rid],sources[0].iloc[lid],sources[1].iloc[curr_rid]))
        return triangles


def explainSamples(dataset,sources,model,originalClass,maxLenAttributeSet):
        ## we suppose that the sample is always on the left source
        attributes = [col for col in list(sources[0]) if col not in ['id']]
        allTriangles = getPositiveCandidates(dataset,sources)
        rankings = []
        triangleIds = []
        flippedPredictions = []
        notFlipped = []
        for triangle in tqdm(allTriangles):
            triangleIds.append((triangle[0].id,triangle[1].id,triangle[2].id))
            if originalClass == 1:
                currentPerturbations,currPerturbedAttr = createPositiveTriangle(triangle[0],triangle[1],triangle[2],\
                                                                                attributes,maxLenAttributeSet)
            else:
                currentPerturbations,currPerturbedAttr = createNegativeTriangle(triangle[0],triangle[1],triangle[2],\
                                                                                attributes,maxLenAttributeSet)
            perturbations_df = currentPerturbations.reset_index(drop=True)
            r1 = triangle[0]
            r1_copy = [r1]*len(perturbations_df)
            r1_df = pd.DataFrame(r1_copy)
            r1_df = r1_df.reset_index(drop=True)
            _renameColumnsWithPrefix('ltable_',r1_df)
            _renameColumnsWithPrefix('rtable_',perturbations_df)
            allPerturbations = pd.concat([r1_df,perturbations_df], axis=1, sort=False)
            allPerturbations['id'] = np.arange(len(allPerturbations))
            allPerturbations['alteredAttribute'] = currPerturbedAttr
            predictions = wrapDm(allPerturbations,model,\
                                  ignore_columns=['ltable_id','rtable_id','alteredAttribute'],batch_size=1)
            curr_flippedPredictions = allPerturbations[(predictions[:,originalClass] <0.5)]
            currNotFlipped = allPerturbations[(predictions[:,originalClass] >0.5)]
            notFlipped.append(currNotFlipped)
            flippedPredictions.append(curr_flippedPredictions)
            ranking = getAttributeRanking(predictions,currPerturbedAttr,originalClass)
            rankings.append(ranking)
        flippedPredictions_df = pd.concat(flippedPredictions,ignore_index=True)
        notFlipped_df = pd.concat(notFlipped,ignore_index=True)
        return rankings,triangleIds,flippedPredictions_df

    
##check if s1 is not superset of one element in s2list 
def _isNotSuperset(s1,s2_list):
    for s2 in s2_list:
        if set(s2).issubset(set(s1)):
            return False
    return True


def getAttributeRanking(proba,alteredAttributes,originalClass):
    attributeRanking = {}
    for i,prob in enumerate(proba):
        if prob[originalClass] <0.5:
            if _isNotSuperset(alteredAttributes[i],list(attributeRanking.keys())):
                attributeRanking[alteredAttributes[i]] = 1
    return attributeRanking
    