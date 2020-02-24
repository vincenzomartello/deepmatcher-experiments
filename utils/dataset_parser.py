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
        dataset = pd.merge(pdata,df_tableB,how='inner',left_on='rtable_id',right_on=right_prefix+'id')
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


def generate_pos_neg_datasets(splits):
    allSamples = pd.concat(splits,ignore_index=True)
    positives_df = allSamples[allSamples.label==1]
    negatives_df = allSamples[allSamples.label==0]
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


def getFullDataset(splits):
    return pd.concat(splits,ignore_index=True)
