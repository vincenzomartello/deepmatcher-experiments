from utils.intermediate_layer_extraction import return_layer_input
from utils.ri_calculator import computeRi
from utils.deepmatcher_utils import wrapDm
from utils.dataset_parser import generate_train_valid_test
import pandas as pd
import os
from utils.mojito2 import prepareDataset,explainSamples


class Explainer:
    def __init__(self,model,attributes):
        self.model = model
        self.attributes = attributes

    
    def getRankingsWhiteBox(self,dataset_dir,dataset_filename,true_label,aggregation_type):
        print('Computing vectors in the classifier space')
        vectors = return_layer_input(self.model,self.model.classifier,dataset_dir,dataset_filename,true_label)
        ri,ri_aggregate = computeRi(self.model.classifier,\
            self.attributes,vectors,true_label,aggregation_type=aggregation_type)
        return ri,ri_aggregate,vectors


    def getRankingsBlackBox(self,dataset_dir,dataset_filename,true_label,max_len_attribute_set,augment_test=False):
        tableA = pd.read_csv(os.path.join(dataset_dir,'tableA.csv'))
        tableB = pd.read_csv(os.path.join(dataset_dir,'tableB.csv'))
        train_df,validation_df,test_df = generate_train_valid_test(dataset_dir,['train.csv','valid.csv','test.csv']\
            ,left_prefix='ltable_',right_prefix='rtable_',drop_lrid=False)
        if augment_test:
            augmented_test = pd.read_csv(os.path.join(dataset_dir,dataset_filename))
        else:
            augmented_test = test_df
        predictions = wrapDm(augmented_test,self.model,ignore_columns=['ltable_id','rtable_id','id','label'])
        tp_group = augmented_test[(predictions[:,1]>=0.5)& (augmented_test['label'] == 1)]
        tn_group = augmented_test[(predictions[:,0] >=0.5)& (augmented_test['label']==0)]
        correctPredictions = pd.concat([tp_group,tn_group])
        rankings,triangles,flipped,notFlipped = explainSamples(correctPredictions,[tableA,tableB],\
            self.model,originalClass=true_label,maxLenAttributeSet=max_len_attribute_set)
        return rankings,flipped