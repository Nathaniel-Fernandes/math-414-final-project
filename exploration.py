# %%
import os
import subprocess
import pandas as pd
import shutil
import matplotlib.pyplot as plt

os.makedirs("exploration/benign", exist_ok=True)
os.makedirs("exploration/malicious", exist_ok=True)

malware_df = pd.read_csv("data/DikeDataset/labels/malware.csv")
malware_df["benign_or_malware"] = "malware"

benign_df = pd.read_csv("data/DikeDataset/labels/benign.csv")
benign_df["benign_or_malware"] = "benign"
benign_df = benign_df.astype(malware_df.dtypes.to_dict())

benign_df.dtypes

labels = pd.concat([benign_df, malware_df]).reset_index(drop=True)

# %%
# for name in labels.loc[benign_df.shape[0]:benign_df.shape[0]*2]['hash']:
#     shutil.copyfile(f'data/DikeDataset/files/malware/{name}.exe', f'deepreflect/data/malicious_unpacked/malware/{name}.exe')


# %%
# # ensure that there's an equal number of benign & malicious examples
# for i in range(benign_df.shape[0], benign_df.shape[0]*2):
#     prefix = 'data/DikeDataset/files/'
#     filepath = f'{labels.iat[i, 12]}/{labels.iat[i, 1]}'
#     postfix = '.exe'

#     subprocess.run(['python', 'main.py', '-o', f'exploration/{filepath}', prefix + filepath + postfix])

# %%
import pandas as pd
import seaborn as sns

# df = pd.read_csv('runs/malwaredetection_2024-04-30 11:01:59.292519.txt')

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# cm = confusion_matrix(df['truth'].tolist(), df['pred'].tolist(), labels=['benign', 'malware'])
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['benign', 'malware'])
# disp.plot()
# plt.title('Confusion Matrix for "malware-dection"')
# # sns.heatmap(cm, fmt='d', annot=True, square=True,
#             cmap='gray_r', vmin=0, vmax=0,  # set all to white
#             linewidths=0.5, linecolor='k',  # draw black grid lines
#             cbar=False)                     # disable colorbar

# # re-enable outer spines
# sns.despine(left=False, right=False, top=False, bottom=False)

# %%
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

# %%
df_results
df_results.groupby("type").agg(["mean", "sem"])
