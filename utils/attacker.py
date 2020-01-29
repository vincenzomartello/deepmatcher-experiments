import random as rd
import string
import pandas as pd
import numpy as np
import deepmatcher as dm
import os
import shutil
from tqdm import tqdm


def _wrapDm(test_df,model,ignore_columns=['id','label'],outputAttributes=False):
    if not os.path.exists('temp'):
        os.mkdir('temp')
    test_df.to_csv('temp/test.csv',index=False)
    test = dm.data.process_unlabeled('temp/test.csv',model,ignore_columns=ignore_columns)
    predictions = model.run_prediction(test, output_attributes=outputAttributes)
    return predictions


def changeRandomCharacters(att,editdist=1):
    alreadyUsedPositions = []
    res = att
    if editdist >len(att):
        editdist = len(att)
    for i in range(editdist):
        random_char = rd.choice(string.ascii_letters+string.digits+string.punctuation)
        new_char_idx = rd.choice(np.arange(len(att)))
        while new_char_idx in alreadyUsedPositions:
            new_char_idx = rd.choice(np.arange(len(att)))
        alreadyUsedPositions.append(new_char_idx)
        res = res.replace(att[new_char_idx],random_char,1)
    return res


def attackDatasetEditDist(dataset,ignore_columns,editdist=1):
    for col in list(dataset):
        if col not in ignore_columns:
            dataset[col] = dataset[col].apply(lambda s: changeRandomCharacters(str(s),editdist))
    return dataset


def cleanString(string,filters = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'):
    for c in filters:
        string = string.replace(c,' ')
    return string


def getSentences(df,ignore_columns=['id','label']):
    sentences = []
    df_index = df.index
    for idx in df_index:
        text = []
        for col in list(df):
            if col not in ignore_columns:
                text.append(str(df.at[idx,col]))
        sentences.append(" ".join(text))
    sentences = list(map(lambda s:cleanString(s),sentences))
    return sentences


def getDictionary(df_l):
    all_sentences = getSentences(pd.concat(df_l,ignore_index=True))
    wordDict = {}
    for sentence in all_sentences:
        for token in sentence.split():
            token = token.lower()
            if token in wordDict:
                wordDict[token] += 1
            else:
                wordDict[token] = 1
    return wordDict


def perturbAttribute(sample,attribute,tokenToReplace,closestWord):
    attr = sample[attribute]
    sample[attribute] = str(attr).replace(tokenToReplace,closestWord)
    return sample


def changeCharactersInAttr(sample,attribute,tokenToAlter,editDist):
    alreadyUsedIdx = []
    alteredToken = tokenToAlter
    rd.seed(2)
    for i in range(editDist):
        random_char = rd.choice(string.ascii_letters+string.digits+string.punctuation)
        new_char_idx = rd.choice(np.arange(len(tokenToAlter)))
        while(new_char_idx in alreadyUsedIdx):
            new_char_idx = rd.choice(np.arange(len(tokenToAlter)))
        alreadyUsedIdx.append(new_char_idx)
        alteredToken = alteredToken.replace(alteredToken[new_char_idx],random_char)
    sample[attribute] = str(sample[attribute]).replace(tokenToAlter,alteredToken)
    return sample


def attackAttribute(dataset,sample_idx,attribute,closestWordsMap,notfound,stopwords):
    attr = dataset.iloc[sample_idx][attribute]
    cleanAttr = cleanString(str(attr))
    attackedTuples = []
    for token in cleanAttr.split():
        if (token.lower() not in stopwords) and (token.lower() not in notfound):
            closestWords = closestWordsMap[token.lower()]
            if len(closestWords) > 5:
                closestWords = closestWords[:5]
            for word,dist in closestWords:
                attackedTuple = perturbAttribute(dataset.iloc[sample_idx].copy(),attribute,token,word)
                attackedTuples.append(attackedTuple)
        elif token.lower() in notfound:
            attackedTuple = changeCharactersInAttr(dataset.iloc[sample_idx].copy(),attribute,token,editDist=1)
            attackedTuples.append(attackedTuple)
    attackedTestDf = pd.DataFrame(data=attackedTuples)
    return attackedTestDf


def attackSample(dataset,sample_idx,attributes,closestWordsMap,notfound,stopwords):
    attackedTuples = []
    alteredAttributes = []
    for att in attributes:
        attackedTuplesOnAttribute = attackAttribute(dataset,sample_idx,att,closestWordsMap,
                                                   notfound,stopwords)
        alteredAttributes= alteredAttributes + ([att]*len(attackedTuplesOnAttribute))
        attackedTuples.append(attackedTuplesOnAttribute)
    allAttacks = pd.concat(attackedTuples)
    allAttacks['altered_attribute'] = alteredAttributes
    return allAttacks


def _check(originalPred,attackedPredictions,attribute,attack_ids):
    attackSuccessfull = {}
    for att in attribute:
        attackSuccessfull[att] = False
    selectedRows = attackedPredictions[attack_ids[0]:attack_ids[-1]+1]
    for i in range(len(selectedRows)):
        if round(selectedRows.iloc[i]['match_score'])!= originalPred:
            attackSuccessfull[selectedRows.iloc[i]['altered_attribute']] = True
    return attackSuccessfull


def attackDataset(dataset,model,attributes,closestWordsMap,notfound,stopwords):
    attackedPairs = []
    j = 0
    ##for which tuple I save the mapping between tuple and its relative attacked records
    pairAttackMapping = {}
    for idx in tqdm(range(len(dataset))):
        curr_attack = attackSample(dataset,idx,attributes,model,closestWordsMap,
                                         notfound,stopwords)
        pairAttackMapping[idx] = np.arange(j,j+len(curr_attack))
        curr_attack['id'] = np.arange(j,j+len(curr_attack))
        j += len(curr_attack)
        attackedPairs.append(curr_attack)
    attackedPairs_df = pd.concat(attackedPairs)
    originalPreds = _wrapDm(dataset,model)
    attackPredictions =_wrapDm(attackedPairs_df,model,ignore_columns=['id','label','altered_attribute'],outputAttributes=True)
    attackSuccessfull = {}
    for i in tqdm(range(len(originalPreds))):
        originalPred = round(originalPreds.iloc[i]['match_score'])
        correspondingAttackRows = pairAttackMapping[i]
        attackSuccessfull[i] = _check(originalPred,attackPredictions,attributes,correspondingAttackRows)
    attackStats_df = pd.DataFrame.from_dict(attackSuccessfull,orient='index')
    shutil.rmtree('temp')
    return attackStats_df


def getAttackedDataset(dataset,attributes,closestWordsMap,notfound,stopwords):
    attackedPairs = []
    j = 0
    ##for which tuple I save the mapping between tuple and its relative attacked records
    pairAttackMapping = {}
    pairPerturbedAttributes = {}
    for idx in tqdm(range(len(dataset))):
        curr_attack = attackSample(dataset,idx,attributes,closestWordsMap,
                                         notfound,stopwords)
        pairAttackMapping[dataset.iloc[idx]['id']] = np.arange(j,j+len(curr_attack))
        pairPerturbedAttributes[dataset.iloc[idx]['id']] = curr_attack['altered_attribute']
        curr_attack['id'] = np.arange(j,j+len(curr_attack))
        j += len(curr_attack)
        attackedPairs.append(curr_attack)
    attackedPairs_df = pd.concat(attackedPairs)
    return attackedPairs_df,pairAttackMapping,pairPerturbedAttributes
