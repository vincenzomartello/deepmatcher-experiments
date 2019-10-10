import pandas as pd
import os
import random

def generate_train_valid_test(dataset_dir,left_prefix,right_prefix):
	df_tableA = pd.read_csv(os.path.join(dataset_dir,'tableA.csv'))
	df_tableB = pd.read_csv(os.path.join(dataset_dir,'tableB.csv'))
	df_train = pd.read_csv(os.path.join(dataset_dir,'train.csv'))
	df_valid = pd.read_csv(os.path.join(dataset_dir,'valid.csv'))
	df_test = pd.read_csv(os.path.join(dataset_dir,'test.csv'))
	left_columns = []
	right_columns = []
	for lcol,rcol in zip(list(df_tableA),list(df_tableB)):
		left_columns.append(left_prefix+lcol)
		right_columns.append(right_prefix+rcol)
	df_tableA.columns = left_columns
	df_tableB.columns = right_columns

	#P sta per parziale
	ptrain=pd.merge(df_train,df_tableA, how='inner')
	pvalid=pd.merge(df_valid,df_tableA,how='inner')
	ptest=pd.merge(df_test,df_tableA,how='inner')
	train=pd.merge(ptrain,df_tableB,how='inner')
	valid=pd.merge(pvalid,df_tableB,how='inner')
	test=pd.merge(ptest,df_tableB,how='inner')

	train_lenght =train.shape[0]
	valid_lenght = valid.shape[0]
	test_lenght = test.shape[0]

	train_ids = random.sample(range(0,train_lenght),train_lenght)
	valid_ids = random.sample(range(train_lenght+100,train_lenght+valid_lenght+100),valid_lenght)
	test_ids = random.sample(range(train_lenght+valid_lenght+200,
		train_lenght+valid_lenght+test_lenght+200),test_lenght)

	train = train.drop(['ltable_id','rtable_id'],axis=1)
	valid = valid.drop(['ltable_id','rtable_id'],axis=1)
	test = test.drop(['ltable_id','rtable_id'],axis=1)

	train['id'] =train_ids
	valid['id'] = valid_ids
	test['id'] = test_ids
	return train,valid,test


def generate_pos_neg_datasets(dataset_dir,train_name,valid_name,test_name):
    train_df = pd.read_csv(os.path.join(dataset_dir,train_name))
    validation_df = pd.read_csv(os.path.join(dataset_dir,valid_name))
    test_df = pd.read_csv(os.path.join(dataset_dir,test_name))
    
    train_negatives = train_df[train_df['label']==0]
    validation_negatives = validation_df[validation_df['label']==0]
    test_negatives = test_df[test_df['label']==0]
    
    train_positives = train_df[train_df['label']==1]
    validation_positives = validation_df[validation_df['label']==1]
    test_positives = test_df[test_df['label']==1]
    
    negatives_df = train_negatives.append(validation_negatives,ignore_index=True)
    negatives_df = negatives_df.append(test_negatives,ignore_index=True)
    
    positives_df = train_positives.append(validation_positives,ignore_index=True)
    positives_df = positives_df.append(test_positives,ignore_index=True)
    
    return (positives_df,negatives_df)