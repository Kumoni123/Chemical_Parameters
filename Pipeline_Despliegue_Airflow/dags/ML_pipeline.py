from airflow.models import DAG
from datetime import datetime, timedelta
from airflow.operators.python import PythonOperator
import pandas as pd
import os
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.preprocessing import PowerTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix

default_args = {
    'owner':'Belyeud',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

def preparation_data():
    import pandas as pd
    import numpy as np
    dataset = pd.read_csv('data/Train.csv', index_col=[0],parse_dates=[0])
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

    df_new.to_csv('data/data_final.csv', index = True)

def division():
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    df_new = pd.read_csv('data/data_final.csv', index_col=[0],parse_dates=[0])
    X = df_new.iloc[:, :-1].values
    y = df_new.iloc[:, 2].values
    split_test_size = 0.2
    split_random_state = 42
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = split_test_size, random_state = split_random_state)
    np.save('files/X_train.npy', X_train)
    np.save('files/X_test.npy', X_test)
    np.save('files/y_train.npy', y_train)
    np.save('files/y_test.npy', y_test)

def training_knn():
    import pandas as pd
    import numpy as np
    from sklearn.neighbors import KNeighborsClassifier

    X_train = np.load('files/X_train.npy', allow_pickle=True)
    y_train = np.load('files/y_train.npy', allow_pickle=True)

    model = KNeighborsClassifier(n_neighbors = 5, metric = "minkowski", p=2)
    model.fit(X_train, y_train)

    import pickle
    with open('files/model.pkl', 'wb') as f:
        pickle.dump(model, f)

def predict():
    import pandas as pd
    import numpy as np
    import pickle
    
    with open('files/model.pkl', 'rb') as f:
        model_knn = pickle.load(f)

    X_test = np.load('files/X_test.npy', allow_pickle=True)
    y_pred = model_knn.predict(X_test)
    np.save('files/y_pred.npy', y_pred)

def metrics():
    import pandas as pd
    import numpy as np
    from sklearn.metrics import accuracy_score, precision_score, recall_score
    from sklearn import metrics

    y_test = np.load('files/y_test.npy', allow_pickle=True)
    y_pred = np.load('files/y_pred.npy', allow_pickle=True)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    print(metrics.classification_report(y_test, y_pred))

with DAG(
    'DAG_ETL_Pipeline_ML',
    default_args = default_args,
    description = 'Creacion de DAG ETL DAG_ETL_Pipeline_ML',
    schedule_interval = timedelta(minutes=5),
    tags = ['ETL', 'Ingenieria']

) as dag:

    get_preparation_data = PythonOperator(
        task_id = 'get_preparation_data',
        python_callable = preparation_data
    )
    
    get_division = PythonOperator(
        task_id = 'get_division',
        python_callable = division
    )

    get_training_knn = PythonOperator(
        task_id = 'get_training_knn',
        python_callable = training_knn
    )

    get_predict = PythonOperator(
        task_id = 'get_predict',
        python_callable = 'predict',
    )

    get_metrics = PythonOperator(
        task_id = 'get_metrics',
        python_callable = metrics
    )

    get_preparation_data >> get_division >> get_training_knn >> get_predict >> get_metrics