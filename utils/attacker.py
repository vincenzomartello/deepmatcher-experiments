import random as rd
import string
import pandas as pd
import numpy as np
import deepmatcher as dm
import os
import shutil


def _wrapDm(test_df,model,ignore_columns=['id','label']):
    if not os.path.exists('temp'):
        os.mkdir('temp')
    test_df.to_csv('temp/test.csv',index=False)
    test = dm.data.process_unlabeled('temp/test.csv',model,ignore_columns=ignore_columns)
    predictions = model.run_prediction(test)
    prediction_matchscore = predictions['match_score']
    predicted_labels = list(map(lambda p:round(p),prediction_matchscore))
    return prediction_matchscore


def changeRandomCharacter(att,permittedPositions):
    random_char = rd.choice(string.ascii_letters+string.digits+string.punctuation)
    new_char_idx = rd.choice(permittedPositions)
    res = att[:new_char_idx]+random_char+att[new_char_idx+1:]
    return res


def attackDataset(dataset,ignore_columns):
    for col in list(dataset):
        if col not in ignore_columns:
            dataset[col] = dataset[col].apply(lambda s: changeRandomCharacter(str(s),
                                                                             np.arange(len(str(s)))))
    return dataset


def cleanString(string,filters = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'):
    for c in filters:
        string = string.replace(c,' ')
    return string


def createDictionary(sentences):
    wordDict = {}
    for sentence in sentences:
        for token in sentence.split():
            token = token.lower()
            if token in wordDict:
                wordDict[token] += 1
            else:
                wordDict[token] = 1
    return wordDict


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
    dictionary = createDictionary(all_sentences)
    return dictionary


def perturbAttribute(sample,attribute,tokenToReplace,closestWord):
    attr = sample[attribute]
    sample[attribute] = str(attr).replace(tokenToReplace,closestWord)
    return sample


def changeCharactersInAttr(sample,attribute,tokenToAlter,editDist):
    alreadyUsedIdx = []
    alteredToken = tokenToAlter
    for i in range(editDist):
        random_char = rd.choice(string.ascii_letters+string.digits+string.punctuation)
        new_char_idx = rd.choice(np.arange(len(tokenToAlter)))
        while(new_char_idx in alreadyUsedIdx):
            new_char_idx = rd.choice(np.arange(len(tokenToAlter)))
        alreadyUsedIdx.append(new_char_idx)
        alteredToken = alteredToken.replace(alteredToken[new_char_idx],random_char)
    sample[attribute] = str(sample[attribute]).replace(tokenToAlter,alteredToken)
    return sample


def attackAttribute(dataset,sample_idx,attribute,model,closestWordsMap,notfound,stopwords):
    attr = dataset.at[sample_idx,attribute]
    cleanAttr = cleanString(str(attr))
    attackedTuples = []
    for token in cleanAttr.split():
        if (token.lower() not in stopwords) and (token.lower() not in notfound):
            closestWords = closestWordsMap[token.lower()]
            for word,dist in closestWords:
                attackedTuple = perturbAttribute(dataset.iloc[sample_idx].copy(),attribute,token,word)
                attackedTuples.append(attackedTuple)
        elif token.lower() in notfound:
            attackedTuple = changeCharactersInAttr(dataset.iloc[sample_idx].copy(),attribute,token,editDist=1)
            attackedTuples.append(attackedTuple)
    attackedTestDf = pd.DataFrame(data=attackedTuples)
    return attackedTestDf


def _findIndex(idx,indexMap):
    for key,val in indexMap.items():
        if idx in val:
            return key
    return None


def attackSample(dataset,sample_idx,attributes,model,closestWordsMap,notfound,stopwords):
    originalPred = round(_wrapDm(dataset.iloc[[sample_idx]],model)[0])
    attackedTuples = []
    perturbationLen = {}
    j = 0
    attackSuccessfullForAttr = {}
    for att in attributes:
        attackedTuplesOnAttribute = attackAttribute(dataset,sample_idx,att,model,closestWordsMap,
                                                   notfound,stopwords)
        perturbationLen[att] = np.arange(j,j+len(attackedTuplesOnAttribute))
        j += len(attackedTuplesOnAttribute)
        attackSuccessfullForAttr[att] = False
        attackedTuples.append(attackedTuplesOnAttribute)
    allAttacks = pd.concat(attackedTuples)
    predictions = _wrapDm(allAttacks,model)
    shutil.rmtree('temp')
    ## boolean array
    for i,pred in enumerate(predictions):
        if round(pred) !=originalPred:
            relatedAttribute = _findIndex(i,perturbationLen)
            attackSuccessfullForAttr[relatedAttribute] = True
    return attackSuccessfullForAttr