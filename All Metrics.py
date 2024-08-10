# All Metrics
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_auc_score,f1_score, average_precision_score,matthews_corrcoef

def calculate_metrics(y_true, y_pred, y_prob):
    acc = accuracy_score(y_true, y_pred)
    sen = recall_score(y_true, y_pred)
    spe = recall_score(y_true, y_pred, pos_label=0)
    auc = roc_auc_score(y_true, y_prob)
    precision = precision_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    aupr = average_precision_score(y_true, y_prob)
    return acc, sen, spe,precision,f1,mcc, auc, aupr

from scipy.stats import ttest_ind
def generate_table(model_res_file,model_names,optimal_idx_=None):
    results=[]
    pre_metrics=None
    t=None
    p=None
    for index,(model_file, model_name) in enumerate(zip(model_res_file,model_names)):
        # v2=pd.read_csv(model_file,header=None).values.flatten()
        if "loocv" in model_file:
            v2=np.nan_to_num(pd.read_csv(model_file,index_col=0).values.flatten())
        else:
            v2=np.nan_to_num(pd.read_csv(model_file,header=None).values.flatten())
        fpr, tpr, thresholds = roc_curve(v1, v2)

        if optimal_idx_ is None:
            # Calculate Youden's J statistic for each threshold
            youden_j = tpr - fpr
            # Find the index of the maximum Youden's J statistic
            optimal_idx = np.argmax(youden_j)
        else:
            optimal_idx = optimal_idx_
        # print(model_name," : ",optimal_idx,len(fpr),len(thresholds))
            # Get the optimal threshold
        
        optimal_threshold = thresholds[optimal_idx]
        preds = deepcopy(v2)
        preds[preds<optimal_threshold]=0
        preds[preds>0]=1
        metrics = calculate_metrics(v1, preds, v2)
        # print(pre_metrics)
        if pre_metrics is not None:
            t,p = ttest_ind(metrics,pre_metrics)
            # print(t,p)
        else:
            pre_metrics = metrics
    
        results_item = pd.DataFrame({
            'Method':model_name,
            "ACC(%)":metrics[0],
            'SEN(%)':metrics[1],
            'SPE(%)':metrics[2],
            'PRE(%)':metrics[3],
            'F1(%)':metrics[4], 
            'MCC(%)':metrics[5],
            'AUC(%)':metrics[6],
            'AUCR(%)':metrics[7], 
            'ttest': t,
            "p":p
        },index=[index])
        results.append(results_item)
    return pd.concat(results)