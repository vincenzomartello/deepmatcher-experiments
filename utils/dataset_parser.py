import pandas as pd
import os
import random
import numpy as np


def generate_train_valid_test(dataset_dir,splitfiles,left_prefix,right_prefix,drop_lrid=True):
    df_tableA = pd.read_csv(os.path.join(dataset_dir,'tableA.csv'))
    df_tableB = pd.read_csv(os.path.join(dataset_dir,'tableB.csv'))
    datasets_ids = []
    for splitname in splitfiles:
        datasets_ids.append(pd.read_csv(os.path.join(dataset_dir,splitname)))
    left_columns = []
    right_columns = []
    for lcol,rcol in zip(list(df_tableA),list(df_tableB)):
        left_columns.append(left_prefix+lcol)
        right_columns.append(right_prefix+rcol)
    df_tableA.columns = left_columns
    df_tableB.columns = right_columns
    datasets = []
    #P sta per parziale
    for dataset_id in datasets_ids:
        pdata = pd.merge(dataset_id,df_tableA, how='inner',left_on='ltable_id',right_on=left_prefix+'id')
        dataset = pd.merge(pdata,df_tableB,how='inner',left_on='rtable_id','right_on=right_prefix+'id')
        datasets.append(dataset)

    train_lenght = datasets[0].shape[0]
    valid_lenght = datasets[1].shape[0]
    test_lenght = datasets[2].shape[0]

    train_ids = np.arange(0,train_lenght)
    valid_ids = np.arange(train_lenght,train_lenght+valid_lenght)
    test_ids = np.arange(train_lenght+valid_lenght,train_lenght+valid_lenght+test_lenght)
    if drop_lrid:
        train = datasets[0].drop(['ltable_id','rtable_id'],axis=1)
        valid = datasets[1].drop(['ltable_id','rtable_id'],axis=1)
        test = datasets[2].drop(['ltable_id','rtable_id'],axis=1)
    else:
        train = datasets[0]
        valid = datasets[1]
        test = datasets[2]
    train['id'] = train_ids
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


def generate_unlabeled(dataset_dir,unlabeled_filename,left_prefix='ltable_',right_prefix='rtable_'):
    df_tableA = pd.read_csv(os.path.join(dataset_dir,'tableA.csv'))
    df_tableB = pd.read_csv(os.path.join(dataset_dir,'tableB.csv'))
    unlabeled_ids = pd.read_csv(os.path.join(dataset_dir,unlabeled_filename))
    left_columns = list(map(lambda s:left_prefix+s,list(df_tableA)))
    right_columns = list(map(lambda s:right_prefix+s,list(df_tableB)))
    df_tableA.columns = left_columns
    df_tableB.columns = right_columns

    #P sta per parziale
    punlabeled = pd.merge(unlabeled_ids,df_tableA, how='inner')
    unlabeled_df = pd.merge(punlabeled,df_tableB,how='inner')

    unlabeled_df = unlabeled_df.drop(['ltable_id','rtable_id'],axis=1)
    unlabeled_df['id'] = np.arange(unlabeled_df.shape[0])
    return unlabeled_df


def getFullDataset(dataset_dir,splitfiles,left_prefix,right_prefix):
    df_tableA = pd.read_csv(os.path.join(dataset_dir,'tableA.csv'))
    df_tableB = pd.read_csv(os.path.join(dataset_dir,'tableB.csv'))
    dataset_splits = []
    for splitname in splitfiles:
        dataset_splits.append(pd.read_csv(os.path.join(dataset_dir,splitname)))
    left_columns = []
    right_columns = []
    for lcol,rcol in zip(list(df_tableA),list(df_tableB)):
        left_columns.append(left_prefix+lcol)
        right_columns.append(right_prefix+rcol)
    df_tableA.columns = left_columns
    df_tableB.columns = right_columns
    allData = pd.concat(dataset_splits)
    #P sta per parziale
    pdata =pd.merge(allData,df_tableA, how='inner')
    dataset_df =pd.merge(pdata,df_tableB,how='inner')
    dataset_df['id'] = np.arange(len(dataset_df))
    return dataset_df
