def accuracy():
    pass


def confusion_matrix():
    pass


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
from classifier import label_2_encoding


def my_confusion_matrix(df):
    """assumes df has columns "true", "pred" that are both str of either {'malware', 'benign'} + "raw_pred" which is the raw output of selection"""

    true = df["true"].tolist()
    pred = df["pred"].tolist()
    cm = confusion_matrix(true, pred, labels=["malware", "benign"])
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=["malware", "benign"]
    )

    return cm, disp


def my_metrics(df):
    """assumes df has columns "true", "pred" that are both str of either {'malware', 'benign'} + "raw_pred" which is the raw output of selection"""

    true = df["true"].tolist()
    pred = df["pred"].tolist()

    accuracy = metrics.accuracy_score(true, pred)
    f1_score = metrics.f1_score(true, pred, pos_label="malware")

    tp, fn, fp, tn = my_confusion_matrix(df)[0].ravel()
    fnr = fn / (tp + fn)

    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    return accuracy, f1_score, fnr, sensitivity, specificity


def my_auc_roc(df):
    """assumes df has columns "true", "pred" that are both str of either {'malware', 'benign'} + "raw_pred" which is the raw output of selection"""

    true = [label_2_encoding[e] for e in df["true"].tolist()]
    pred = df["pred"].tolist()
    pred_raw = df["raw_pred"].tolist()

    fpr, tpr, thresholds = metrics.roc_curve(
        true, pred_raw, pos_label=label_2_encoding["malware"]
    )
    roc_auc = metrics.auc(fpr, tpr)
    return metrics.roc_auc_score(true, pred_raw), metrics.RocCurveDisplay(
        fpr=fpr, tpr=tpr, roc_auc=roc_auc
    )


df_results = pd.DataFrame(
    {
        "accuracy": [],
        "f1_score": [],
        "fnr": [],
        "auroc_score": [],
        "sensitivity": [],
        "specificity": [],
        "type": [],
    }
)

for csv in os.listdir("runs"):
    df = pd.read_csv(f"runs/{csv}", index_col=0)

    cm, cm_plot = my_confusion_matrix(df)
    accuracy, f1_score, fnr, sensitivity, specificity = my_metrics(df)
    auroc_score, roc_plot = my_auc_roc(df)

    df_results.loc[len(df_results)] = [
        accuracy,
        f1_score,
        fnr,
        sensitivity,
        specificity,
        auroc_score,
        "gnb" if "gnb" in csv else "malware-dection",
    ]
