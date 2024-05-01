import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score, matthews_corrcoef, 
    accuracy_score, roc_auc_score, confusion_matrix
)


def eval(y_true, y_pred, model_name="", verbose=True):
    y_true = np.array(y_true).reshape(-1)
    y_pred = np.array(y_pred).reshape(-1)
    
    if ((y_pred>0) & (y_pred<1)).sum()>0:
        y_pred_binary = (y_pred > 0.5).astype(float)
    else:
        y_pred_binary = y_pred
    
    accuracy = accuracy_score(y_true, y_pred_binary)
    roc_auc = roc_auc_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred_binary)
    recall = recall_score(y_true, y_pred_binary)
    f1 = f1_score(y_true, y_pred_binary)
    mcc = matthews_corrcoef(y_true, y_pred_binary)

    # Calculate confusion matrix and extract true negatives (tn), false positives (fp), false negatives (fn), and true positives (tp)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()

    # Calculate specificity
    specificity = tn / (tn + fp)

    # Display all evaluation metrics
    if verbose:
        print("Accuracy:", accuracy)
        print("ROC AUC Score:", roc_auc)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1 Score:", f1)
        print("Matthews Correlation Coefficient:", mcc)
        print("Specificity (Negative Prediction Accuracy):", specificity)
        
    return {
        "accuracy": accuracy,
        "roc_auc": roc_auc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "matthews_corr": mcc,
        "specificity": specificity,
    }