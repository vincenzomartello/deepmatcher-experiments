import pandas as pd


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
