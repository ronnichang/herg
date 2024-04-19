from sklearn.metrics import (
    precision_score, recall_score, f1_score, matthews_corrcoef, 
    accuracy_score, roc_auc_score, confusion_matrix
)

def eval(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)

    # Calculate confusion matrix and extract true negatives (tn), false positives (fp), false negatives (fn), and true positives (tp)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    # Calculate specificity
    specificity = tn / (tn + fp)

    # Display all evaluation metrics
    print("Accuracy:", accuracy)
    print("ROC AUC Score:", roc_auc)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("Matthews Correlation Coefficient:", mcc)
    print("Specificity (Negative Prediction Accuracy):", specificity)