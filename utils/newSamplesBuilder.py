import pandas as pd
import numpy as np


##new attribute value is a couple of attributes
def buildNewSamples(dataset,selectedAttr,newAttributeVal,newSamples_len,label,left_prefix='ltable_',
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



def buildNewSamplesForAttribute(critical_forPos,critical_forNeg,attribute,lenNewPositives,lenNewNegatives,
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