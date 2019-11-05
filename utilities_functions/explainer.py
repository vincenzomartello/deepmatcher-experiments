import pandas as pd
import deepmatcher as dm
import os


#k is the number of neighbours to consider
def generateExplanations(nearest_neighbours,k,dataset,attribute,model,testset_path,true_label,temp_path='temp'):
    nearest_neighbours_values = pd.merge(nearest_neighbours,dataset,left_on=attribute,right_on='id')
    selected_cols = nearest_neighbours_values[['ltable_'+attribute,'rtable_'+attribute]]
    most_frequents_values = selected_cols.groupby(['ltable_'+attribute,'rtable_'+attribute]).size().reset_index(name='Time').sort_values(by=['Time'],ascending=False).reset_index()
    if (k>= most_frequents_values.shape[0]):
        k = most_frequents_values.shape[0]
    testset = dm.data.process_unlabeled(testset_path,model,ignore_columns=['id','label'])
    standard_preds = model.run_prediction(testset)
    if (true_label == 1):
        true_positives = standard_preds[standard_preds['match_score']>0.5]
    else:
        true_positives = standard_preds[standard_preds['match_score']<=0.5]
    print("The standard true positives are {}".format(str(true_positives.shape[0])))
    for i in range(k):
        testset = pd.read_csv(testset_path)
        testset['ltable_'+attribute] = most_frequents_values.loc[i]['ltable_'+attribute]
        testset['rtable_'+attribute] = most_frequents_values.loc[i]['rtable_'+attribute]
        print(most_frequents_values.loc[i]['ltable_'+attribute]+" | "+most_frequents_values.loc[i]['rtable_'+attribute])
        altered_testName = 'altered_test'+str(i)+'.csv'
        testset.to_csv(os.path.join(temp_path,altered_testName),index=False)
        altered_test = dm.data.process_unlabeled(os.path.join(temp_path,altered_testName),model,ignore_columns = ['id','label'])
        altered_pred = model.run_prediction(altered_test)
        if (true_label == 1):
            true_positives_foraltered = altered_pred[altered_pred['match_score'] >0.5]
        else:
            true_positives_foraltered = altered_pred[altered_pred['match_score']<=0.5]
        print(true_positives_foraltered.shape[0])
    return most_frequents_values


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