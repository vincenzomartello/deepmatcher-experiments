import pandas as pd
import numpy as np
import random as rd


class SampleBuilder:
    def __init__(self,comparator,leftPrefix='ltable_',rightPrefix='rtable_'):
        ##string comparator
        self.comparator = comparator
        self.leftPrefix = leftPrefix
        self.rightPrefix = rightPrefix


    ##df1 and df2 are two sources of records, attributes are the attributes important for the prediction
    def createPossibleMatchings(self,df1,df2,important_attributes,min_similarity,newsample_len):
        df1_index = list(df1.index)
        df2_index = list(df2.index)
        newSamples = []
        while(len(newSamples)<newsample_len):
            current_lrow = df1.iloc[rd.choice(df1.index)]
            current_df = df2
            for att in important_attributes:
                l_att = str(current_lrow[att])
                mask = current_df.apply(lambda row: self.comparator.similarity(str(row[att]),l_att)
                                        >=min_similarity,axis=1)
                current_df = current_df[mask]
            for idx in current_df.index:
                newSamples.append((current_lrow['id'],current_df.at[idx,'id']))
        newSamples_ids = pd.DataFrame(data=newSamples,columns=['ltable_id','rtable_id'])
        ##only temporary label
        newSamples_ids['label'] = np.ones(newSamples_ids.shape[0])
        left_columns = list(map(lambda s:self.leftPrefix+s,list(df1)))
        right_columns = list(map(lambda s:self.rightPrefix+s,list(df2)))
        df1.columns = left_columns
        df2.columns = right_columns

        #P sta per parziale
        punlabeled = pd.merge(newSamples_ids,df1, how='inner')
        unlabeled_df = pd.merge(punlabeled,df2,how='inner')

        unlabeled_df = unlabeled_df.drop(['ltable_id','rtable_id'],axis=1)
        unlabeled_df['id'] = np.arange(unlabeled_df.shape[0])
        return unlabeled_df,newSamples_ids


    ##new attribute value is a couple of attributes
    def buildNewSamples(self,dataset,selectedAttr,newAttributeVal,newSamples_len,label,left_prefix='ltable_',
                       right_prefix='rtable_'):

        new_samples = pd.DataFrame(data = [], columns =list(dataset))
        for i in range(newSamples_len):
            selected_row = dataset.sample()
            if label==0:
                selected_row[left_prefix+selectedAttr] = newAttributeVal[0]
                selected_row[right_prefix+selectedAttr] = newAttributeVal[1]
            else:
                selected_row[left_prefix+selectedAttr] = selected_row[left_prefix+selectedAttr]+" "+newAttributeVal[0]
                selected_row[right_prefix+selectedAttr] = selected_row[right_prefix+selectedAttr]+" "+newAttributeVal[1]
            new_samples = new_samples.append(selected_row,ignore_index=True)
        return new_samples



    def buildNewSamplesForAttribute(self,critical_forPos,critical_forNeg,attribute,lenNewPositives,lenNewNegatives,
                                   start_idx):
        newSamples = []
        for df,_,_ in critical_forPos[attribute]:
            if df.shape[0] < lenNewPositives:
                newSamples.append(df)
            else:
                newSamples.append(df.sample(n= lenNewPositives))
        for df,_,_ in critical_forNeg[attribute]:
            if df.shape[0] < lenNewNegatives:
                newSamples.append(df)
            else:
                newSamples.append(df.sample(n= lenNewNegatives))
        newSamples = pd.concat(newSamples)
        newSamples = newSamples.drop(columns=['match_score'])
        newSamples['id'] = np.arange(start_idx,start_idx+newSamples.shape[0])
        return newSamples