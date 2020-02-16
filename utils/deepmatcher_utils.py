import deepmatcher as dm
import numpy as np
import contextlib
import os
import pandas as pd
import shutil
from tqdm import tqdm
from sklearn.metrics import f1_score,precision_score,recall_score


def wrapDm(test_df,model,ignore_columns=['id','label'],outputAttributes=False,batch_size=32):
    if not os.path.exists('temp'):
        os.mkdir('temp')
    test_df.to_csv('temp/test.csv',index=False)
    with open(os.devnull, 'w') as devnull:
        with contextlib.redirect_stdout(devnull):
            data_processed = dm.data.process_unlabeled('temp/test.csv', trained_model = model,\
                                                       ignore_columns=ignore_columns)
            predictions = model.run_prediction(data_processed, output_attributes = outputAttributes,\
                                              batch_size=batch_size)
            out_proba = predictions['match_score'].values
    multi_proba = np.dstack((1-out_proba, out_proba)).squeeze()
    shutil.rmtree('temp')
    if outputAttributes:
        return predictions
    else:
        return multi_proba



def getF1PrecisionRecall(true_labels,predicted_labels):
    y_pred = np.argmax(predicted_labels,axis=1)
    return (f1_score(true_labels,y_pred),precision_score(true_labels,y_pred),
            recall_score(true_labels,y_pred))


def getMeanConfidenceAndVariance(model,test_df,ignoreColumns=['id','label']):
    predictions = wrapDm(test_df,model,ignore_columns=ignoreColumns)
    confidences = np.amax(predictions,axis=1)
    meanConfidence = sum(confidences)/len(confidences)
    variance = np.var(confidences)
    return meanConfidence,variance


def getTruePositiveNegative(model,df,ignore_columns):
    predictions = wrapDm(df,model,ignore_columns=ignore_columns)
    tp_group = df[(predictions[:,1]>=0.5)& df['label'] == 1]
    tn_group = df[(predictions[:,0] >=0.5)& df['label']==0]
    correctPredictions = pd.concat([tp_group,tn_group])
    return correctPredictions