import pandas as pd
import deepmatcher as dm
import os


class Explainer:
    def __init__(self,k,model,testset_path,temp_path='temp',left_prefix='ltable_',right_prefix='rtable_'):
        self.k = k
        self.model = model
        self.testset_path = testset_path
        self.temp_path = temp_path
        self.left_prefix = left_prefix
        self.right_prefix = right_prefix
        
            
    #explanation nr is the number of neighbors to consider
    def generateExplanations(nearest_neighbors,opposite_label_df,
                             attribute,threshold,true_label):

        nn_frequencies = nearest_neighbors[attribute].value_counts().rename_axis('unique_values').reset_index(name='counts')
        nearest_neighbors_values = pd.merge(nn_frequencies,opposite_label_df,left_on='unique_values',right_on='id')
        selected_cols = nearest_neighbors_values[[self.leftPrefix+attribute,self.rightPrefix+attribute]]
        if (self.k >= selected_cols.shape[0]):
            explanation_nr = selected_cols.shape[0]
        else:
            explanation_nr = self.k
        testset = dm.data.process_unlabeled(self.testset_path,self.model,ignore_columns=['id','label'])
        standard_preds = self.model.run_prediction(testset)
        if (true_label == 1):
            original_false_negatives = standard_preds[standard_preds['match_score'] <= 0.5].shape[0]
        else:
            original_false_negatives = standard_preds[standard_preds['match_score'] > 0.5].shape[0]
        most_frequent_values = selected_cols.head(explanation_nr)
        falseNegatives = []
        for i in range(explanation_nr):
            testset = pd.read_csv(self.testset_path)
            lval = str(most_frequent_values.loc[i][self.leftPrefix+attribute])
            rval = str(most_frequent_values.loc[i][self.rightPrefix+attribute])
            testset[self.leftPrefix+attribute] = lval
            testset[self.rightPrefix+attribute] = rval
            altered_testName = 'altered_test'+str(i)+'.csv'
            testset.to_csv(os.path.join(self.temp_path,altered_testName),index=False)
            altered_test = dm.data.process_unlabeled(os.path.join(self.temp_path,altered_testName),self.model,
                                                     ignore_columns = ['id','label'])
            altered_pred = self.model.run_prediction(altered_test,output_attributes=True)
            if (true_label == 1):
                false_negatives = altered_pred[altered_pred['match_score'] <=0.5]
            else:
                false_negatives = altered_pred[altered_pred['match_score'] > 0.5]
            if ((false_negatives.shape[0]-original_false_negatives)/original_false_negatives) >=threshold:
                ##append critical values and how much false negatives I have generated
                falseNegatives.append((false_negatives,lval,rval))
        return falseNegatives


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

    
    ##Given an attribute pair return the true positives calculated from the model for the attribute pair
    ## if the attribute pair is inserted in place of original values
    def testRobustness(testset_path,attribute,true_label,substitute_values):
        standard_test = dm.data.process_unlabeled(testset_path,self.model,ignore_columns=['id','label'])
        standard_pred = model.run_prediction(standard_test)
        test_df = pd.read_csv(testset_path)
        if true_label == 1:
            original_true_pos = standard_pred[standard_pred.match_score>0.5].shape[0]
        else:
            original_true_pos = standard_pred[standard_pred.match_score<=0.5].shape[0]
        lval = substitute_values[0]
        rval = substitute_values[1]
        test_df['ltable_'+attribute] = lval
        test_df['rtable_'+attribute] = rval
        test_df.to_csv(os.path.join(self.temp_path,'new_test.csv'),index=False)
        new_test = dm.data.process_unlabeled(os.path.join(self.temp_path,'new_test.csv'),self.model
                                             ,ignore_columns=['id','label'])
        new_pred = model.run_prediction(new_test,output_attributes=True)
        if true_label ==1:
            new_true_pos = new_pred[new_pred.match_score >0.5].shape[0]
        else:
            new_true_pos = new_pred[new_pred.match_score <=0.5].shape[0]
        return original_true_pos,new_true_pos