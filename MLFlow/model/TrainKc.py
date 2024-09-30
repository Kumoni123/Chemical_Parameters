# -*- coding: utf-8 -*-
"""
@author: Belyeud
"""

# In[0]: Librerías y Configuraciones
from pathlib import Path

import joblib
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import mlflow
from ydata_profiling import ProfileReport
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix


def create_confusion_matrices(label_encoder, y_train, y_pred_train, y_test, y_pred_test):
    """
    This function creates confusion matrices for the training and testing datasets.

    Parameters:
    label_encoder (LabelEncoder): The LabelEncoder used for encoding the classes.
    y_train (array-like): The true labels for the training set.
    y_pred_train (array-like): The predicted labels for the training set.
    y_test (array-like): The true labels for the testing set.
    y_pred_test (array-like): The predicted labels for the testing set.

    Returns:
    Figure: A matplotlib Figure object with the confusion matrices.
    """
    fig, ax = plt.subplots(nrows=1, ncols=2)
    cf_train = confusion_matrix(y_train, y_pred_train)
    cf_test = confusion_matrix(y_test, y_pred_test)
    sns.heatmap(
        cf_train,
        annot=True,
        annot_kws={"fontsize": 10, "fontweight": "bold"},
        fmt="",
        cmap="Reds",
        cbar=False,
        square=True,
        linewidths=1.1,
        yticklabels=label_encoder.classes_,
        xticklabels=label_encoder.classes_,
        ax=ax[0],
    )
    ax[0].set_yticklabels(ax[0].get_yticklabels(), rotation=0)
    ax[0].set_title("Train", fontsize=12, fontweight="bold", color="red")
    sns.heatmap(
        cf_test,
        annot=True,
        annot_kws={"fontsize": 10, "fontweight": "bold"},
        fmt="",
        cmap="Blues",
        cbar=False,
        square=True,
        linewidths=1.1,
        yticklabels=label_encoder.classes_,
        xticklabels=label_encoder.classes_,
        ax=ax[1],
    )
    ax[1].set_yticklabels(ax[0].get_yticklabels(), rotation=0)
    ax[1].set_title("Test", fontsize=12, fontweight="bold", color="blue")
    fig.suptitle("Confusion Matrix", fontsize=14, fontweight="bold", color="black")
    fig.tight_layout()
    return fig


def plot_feature_importance(feature_importance):
    """
    This function creates a bar plot for feature importances.

    Parameters:
    feature_importance (DataFrame): A DataFrame with 'feature' and 'importance' columns.

    Returns:
    Figure: A matplotlib Figure object with the feature importance plot.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    feature_importance.plot(x="feature", y="importance", kind="bar", ax=ax)
    ax.set_title("Feature Importance")
    ax.set_xlabel("Features")
    ax.set_ylabel("Importance")
    fig.tight_layout()
    return fig


def make_classification_report_frame(
    report,
    classes,
):
    """
    This function transforms the classification report dictionary into a pandas DataFrame.

    Parameters:
    report (dict): The dictionary resulting from calling `classification_report` with `output_dict=True`.
    classes (list): The list of classes used in the classification task.

    Returns:
    DataFrame: A pandas DataFrame where each row corresponds to a class and each column corresponds to a metric.
               The metrics are 'precision', 'recall', 'f1-score', and 'support'.
               The DataFrame also includes rows for 'micro avg', 'macro avg', and 'weighted avg' if they are present in the report.

    Example:
    >>> report = classification_report(y_true, y_pred, output_dict=True)
    >>> classes = ['class1', 'class2']
    >>> df = make_classification_report_frame(report, classes)
    """
    metrics = [
        "precision",
        "recall",
        "f1-score",
        "support",
    ]
    data = []
    for label in classes:
        data.append([label] + [report[label][metric] for metric in metrics])
    general = [
        "micro avg",
        "macro avg",
        "weighted avg",
    ]
    for label in general:
        if label in report and isinstance(report[label], dict):
            data.append([label] + [report[label].get(metric, None) for metric in metrics])
    return pd.DataFrame(data, columns=["label"] + metrics)

output_dir = Path("output")
output_dir.mkdir(parents=True, exist_ok=True)

mlflow.set_experiment("Chemical Parameters - Earlier Contamination Prediction")
mlflow.set_experiment_tags(
    {
        "project": "Chemical Parameters",
        "task": "Classification",
    }
)

run = mlflow.start_run()

# In[1]: Preprocesamiento
dataset = pd.read_csv('model/Train.csv', index_col=[0],parse_dates=[0])

df = dataset.copy()
      
from imblearn.combine import SMOTETomek
df2 = df.copy()
df3 = df2[df2['Falla'] != 'A']

df3 = pd.concat([df3, df2[df2['Falla'] == 'A'].sample(frac=0.45, random_state = 0)])

print(df3['Falla'].value_counts())  


X = df3.iloc[:,0:-1]
y = df3.iloc[:,-1]

os_us = SMOTETomek(sampling_strategy='auto',random_state=0)

X_res, y_res = os_us.fit_resample(X, y)

from collections import Counter
print ("Distribution before resampling {}".format(Counter(y)))
print ("Distribution after resampling {}".format(Counter(y_res)))

df_new = X_res.copy()
df_new['Tipo'] = np.array(y_res)

data_report = ProfileReport(df_new, title="Data Report")
data_report.to_file(output_dir / "data_report.html")
mlflow.log_artifact(output_dir / "data_report.html")

X = df_new.iloc[:, :-1].values
y = df_new.iloc[:, 2].values

# In[2]: División
split_test_size = 0.2
split_random_state = 42

mlflow.log_param("split_test_size", split_test_size)
mlflow.log_param("split_random_state", split_random_state)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = split_test_size, random_state = split_random_state)

mlflow.log_param("x_train_shape", X_train.shape)
mlflow.log_param("x_test_shape", X_test.shape)
mlflow.log_param("y_train_shape", y_train.shape)
mlflow.log_param("y_test_shape", y_test.shape)


from sklearn.preprocessing import MinMaxScaler, LabelEncoder
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)


sc = MinMaxScaler(feature_range=(0,1))
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

from sklearn.preprocessing import PowerTransformer
pt = PowerTransformer(method = 'yeo-johnson', standardize=True)

skl_boxcox_x_train = pt.fit(X_train) #1ra columna
skl_boxcox_x_test = pt.fit(X_test) #1ra columna

X_train = pt.transform(X_train)
X_test = pt.transform(X_test)

X_train_prep = pd.DataFrame(X_train, columns = ['Kc', 'FW_MUESTRA'])
X_test_prep = pd.DataFrame(X_test, columns = ['Kc', 'FW_MUESTRA'])

X_test_prep.to_csv(output_dir / "X_test_prep.csv", index=False)
X_train_prep.to_csv(output_dir / "X_train_prep.csv", index=False)


#In[3]: Entrenamiento Knn
"""
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

#KNN
model_knn = KNeighborsClassifier(n_neighbors = 5, metric = "minkowski", p=2)
model_knn.fit(X_train, y_train)

mlflow.log_params({
    f"knn_{param}": value for param, value in model_knn.get_params().items()
})


joblib.dump(label_encoder, output_dir / "label_encoder.pkl")
mlflow.log_artifact(output_dir / "label_encoder.pkl")
joblib.dump(sc, output_dir / "min_max_scaler.pkl")
mlflow.log_artifact(output_dir / "min_max_scaler.pkl")
joblib.dump(pt, output_dir / "powertransformer.pkl")
mlflow.log_artifact(output_dir / "powertransformer.pkl")
joblib.dump(model_knn, output_dir / "knn_model.pkl")
mlflow.log_artifact(output_dir / "knn_model.pkl")

y_pred_test = model_knn.predict(X_test_prep)
y_pred_train = model_knn.predict(X_train_prep)

test_accuracy = accuracy_score(y_test, y_pred_test)
train_accuracy = accuracy_score(y_train, y_pred_train)

mlflow.log_metric("train_accuracy", train_accuracy)
mlflow.log_metric("test_accuracy", test_accuracy)

# Print classification report
clf_report = classification_report(y_test, y_pred_test, target_names=label_encoder.classes_, output_dict=True)
for label in label_encoder.classes_:
    for metric in clf_report[label]:
        mlflow.log_metric(f"{metric}_{label}", clf_report[label][metric])

report_frame = make_classification_report_frame(clf_report, label_encoder.classes_)

mlflow.log_table(report_frame, "classification_report.json")


# Log feature importances
feature_importance = pd.DataFrame(
    {"feature": X_train_prep.columns, "importance": model_knn.feature_importances_}
).sort_values("importance", ascending=False)

mlflow.log_table(feature_importance, "feature_importance.json")

feature_importance.to_csv(output_dir / "feature_importance.csv", index=False)

feature_importance_fig = plot_feature_importance(feature_importance)
mlflow.log_figure(feature_importance_fig, "feature_importance.png")


# Create confusion matrices for train and test
confusion_matrix_figure = create_confusion_matrices(label_encoder, y_train, y_pred_train, y_test, y_pred_test)
mlflow.log_figure(confusion_matrix_figure, "confusion_matrix.png")


print(f"Experiment ID: {run.info.experiment_id}")
print(f"Run ID: {run.info.run_id}")
"""
#In[4]: Entrenamiento Knn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

#KNN
model_lr = LogisticRegression()
model_lr.fit(X_train, y_train)

mlflow.log_params({
    f"lr_{param}": value for param, value in model_lr.get_params().items()
})


joblib.dump(label_encoder, output_dir / "label_encoder.pkl")
mlflow.log_artifact(output_dir / "label_encoder.pkl")
joblib.dump(sc, output_dir / "min_max_scaler.pkl")
mlflow.log_artifact(output_dir / "min_max_scaler.pkl")
joblib.dump(pt, output_dir / "powertransformer.pkl")
mlflow.log_artifact(output_dir / "powertransformer.pkl")
joblib.dump(model_lr, output_dir / "lr_model.pkl")
mlflow.log_artifact(output_dir / "lr_model.pkl")

y_pred_test = model_lr.predict(X_test_prep)
y_pred_train = model_lr.predict(X_train_prep)

test_accuracy = accuracy_score(y_test, y_pred_test)
train_accuracy = accuracy_score(y_train, y_pred_train)

mlflow.log_metric("train_accuracy", train_accuracy)
mlflow.log_metric("test_accuracy", test_accuracy)

# Print classification report
clf_report = classification_report(y_test, y_pred_test, target_names=label_encoder.classes_, output_dict=True)
for label in label_encoder.classes_:
    for metric in clf_report[label]:
        mlflow.log_metric(f"{metric}_{label}", clf_report[label][metric])

report_frame = make_classification_report_frame(clf_report, label_encoder.classes_)

mlflow.log_table(report_frame, "classification_report.json")

"""
# Log feature importances
feature_importance = pd.DataFrame(
    {"feature": X_train_prep.columns, "importance": model_knn.feature_importances_}
).sort_values("importance", ascending=False)

mlflow.log_table(feature_importance, "feature_importance.json")

feature_importance.to_csv(output_dir / "feature_importance.csv", index=False)

feature_importance_fig = plot_feature_importance(feature_importance)
mlflow.log_figure(feature_importance_fig, "feature_importance.png")
"""

# Create confusion matrices for train and test
confusion_matrix_figure = create_confusion_matrices(label_encoder, y_train, y_pred_train, y_test, y_pred_test)
mlflow.log_figure(confusion_matrix_figure, "confusion_matrix.png")


print(f"Experiment ID: {run.info.experiment_id}")
print(f"Run ID: {run.info.run_id}")