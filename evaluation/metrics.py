from sklearn.metrics import(
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

def calculate_metrics(y_true,y_pred):
    metrics = {
        "accuracy":accuracy_score(y_true,y_pred),
        "precision":precision_score(y_true,y_pred,average="weighted"),
        "recall":recall_score(y_true,y_pred,average="weighted"),
        "f1":f1_score(y_true,y_pred,average="weighted"),
        "report":classification_report(y_true,y_pred,digits=4)
    }
    return metrics