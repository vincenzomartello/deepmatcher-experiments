import pandas as pd
import random as rd


##new attribute value is a couple of attributes
##new attribute value is a couple of attributes
def buildNewSamples(dataset,selectedAttr,newAttributeVal,newSamples_len,left_prefix='ltable_',
                   right_prefix='rtable_'):

    new_samples = pd.DataFrame(data = [], columns =list(dataset))
    for i in range(newSamples_len):
        selected_row = dataset.iloc[rd.randint(0,dataset.shape[0])].to_dict()
        selected_row[left_prefix+selectedAttr] = newAttributeVal[0]
        selected_row[right_prefix+selectedAttr] = newAttributeVal[1]
        new_samples = new_samples.append(selected_row,ignore_index=True)
    return new_samples
