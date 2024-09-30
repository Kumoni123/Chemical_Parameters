from airflow.models import DAG
from datetime import datetime, timedelta
from airflow.operators.python import PythonOperator
from airflow.operators.dummy_operator import DummyOperator
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
    dataset = pd.read_csv('data/new_data.csv', index_col=[0],parse_dates=[0])
    df = dataset.copy()
    df = df.dropna()

    df.to_csv('output/data_final.csv', index = True)

def preprocesamiento():
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.preprocessing import PowerTransformer
    df_new = pd.read_csv('output/data_final.csv', index_col=[0],parse_dates=[0])

    sc = MinMaxScaler(feature_range=(0,1))
    xarray = sc.fit_transform(np.array(df_new))

    pt = PowerTransformer(method = 'yeo-johnson', standardize=True)
    skl_boxcox = pt.fit(xarray) #1ra columna

    xarray = pt.transform(xarray)

    df_out = pd.DataFrame(xarray, columns=['Kc', 'FW_MUESTRA'])

    df_out.to_csv('output/data_final_despliegue.csv', index = False)


def predictions():
    import pandas as pd
    import numpy as np
    import pickle

    df_pred = pd.read_csv('data/new_data.csv', index_col=[0],parse_dates=[0])

    model = pickle.load(open('files/model.pkl', 'rb'))
    y_pred3 = model.predict(df_pred)

    df_pred['Falla'] =  y_pred3

    df_pred.to_csv('output/predictions.csv', index = True)

def drift_detection():
    import pandas as pd
    import numpy as np
    from evidently import ColumnMapping
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset
    from datetime import datetime

    column_mapping = ColumnMapping()
    data_drift = Report(metrics=[DataDriftPreset()])


    new_data = pd.read_csv('output/predictions.csv', index_col=[0],parse_dates=[0])
    old_data = pd.read_csv('data/Train.csv', index_col=[0],parse_dates=[0])

    data_drift_report = Report(metrics=[DataDriftPreset()])
    data_drift_report.run(reference_data=old_data, current_data=new_data)
    report_json = data_drift_report.as_dict()
    drift_detected = report_json["metrics"][0]["result"]["dataset_drift"]

    data_drift.run(current_data=new_data,
                    reference_data=old_data,
                    column_mapping=column_mapping)

    today_date = datetime.today().strftime("%Y-%m-%d")

    if drift_detected == True:
        data_drift.save_html(f"reporte/data_drift_report_{today_date}.html")



with DAG(
    'DAG_ETL_Pipeline_ML',
    default_args = default_args,
    description = 'Creacion de DAG ETL DAG_ETL_Pipeline_ML',
    schedule_interval = timedelta(minutes=5),
    tags = ['ETL', 'Ingenieria']

) as dag:

    get_PI_System = DummyOperator(task_id='get_PI_System')

    get_preparation_data = PythonOperator(
        task_id = 'get_preparation_data',
        python_callable = preparation_data
    )
    
    get_preprocesamiento = PythonOperator(
        task_id = 'get_preprocesamiento',
        python_callable = preprocesamiento
    )

    get_predictions = PythonOperator(
        task_id = 'get_predictions',
        python_callable = predictions
    )

    get_drift_detection = PythonOperator(
        task_id = 'get_drift_detection',
        python_callable = drift_detection
    )

    get_PI_Vision = DummyOperator(task_id='get_PI_Vision')

    get_PI_System >> get_preparation_data >> get_preprocesamiento >> get_predictions >> get_drift_detection >> get_PI_Vision