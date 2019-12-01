import pandas as pd
import deepmatcher as dm
import os

            
#explanation nr is the number of neighbors to consider
def generateExplanations(nearest_neighbors,explanation_nr,threshold,opposite_label_df,
                         attribute,model,testset_path,true_label,temp_path='temp'):
    
    nn_frequencies = nearest_neighbors[attribute].value_counts().rename_axis('unique_values').reset_index(name='counts')
    nearest_neighbors_values = pd.merge(nn_frequencies,opposite_label_df,left_on='unique_values',right_on='id')
    selected_cols = nearest_neighbors_values[['ltable_'+attribute,'rtable_'+attribute]]
    if (explanation_nr >= selected_cols.shape[0]):
        explanation_nr = selected_cols.shape[0]
    testset = dm.data.process_unlabeled(testset_path,model,ignore_columns=['id','label'])
    standard_preds = model.run_prediction(testset)
    if (true_label == 1):
        true_positives = standard_preds[standard_preds['match_score']>0.5]
    else:
        true_positives = standard_preds[standard_preds['match_score']<=0.5]
    most_frequent_values = selected_cols.head(explanation_nr)
    critical_values = []
    for i in range(explanation_nr):
        testset = pd.read_csv(testset_path)
        lval = str(most_frequent_values.loc[i]['ltable_'+attribute])
        rval = str(most_frequent_values.loc[i]['rtable_'+attribute])
        testset['ltable_'+attribute] = lval
        testset['rtable_'+attribute] = rval
        altered_testName = 'altered_test'+str(i)+'.csv'
        testset.to_csv(os.path.join(temp_path,altered_testName),index=False)
        altered_test = dm.data.process_unlabeled(os.path.join(temp_path,altered_testName),model,ignore_columns = ['id','label'])
        altered_pred = model.run_prediction(altered_test)
        if (true_label == 1):
            true_positives_foraltered = altered_pred[altered_pred['match_score'] >0.5]
        else:
            true_positives_foraltered = altered_pred[altered_pred['match_score']<=0.5]
        if ((true_positives.shape[0]-true_positives_foraltered.shape[0])/true_positives.shape[0]) >= threshold:
            critical_values.append((lval,rval))
    return critical_values


## this function return occurrences in negatives and positives samples
def analyze_valueDistribution(dataset,value,attribute):
    dataset_pos = dataset[dataset['label']==1]
    dataset_neg = dataset[dataset['label']==0]
    
    negAttValues = pd.concat([dataset_neg['ltable_'+attribute],dataset_neg['rtable_'+attribute]])
    posAttValues = pd.concat([dataset_pos['ltable_'+attribute],dataset_pos['rtable_'+attribute]])

    posProvenanceDist = dict(negAttValues.value_counts())
    negProvenanceDist = dict(posAttValues.value_counts())
    if value in posProvenanceDist and value in negProvenanceDist:
        return negProvenanceDist[value],posProvenanceDist[value]
    elif value in negProvenanceDist:
        return negProvenanceDist[value],0
    elif value in posProvenanceDist:
        return 0,posProvenanceDist[value]
    else:
        return 0,0