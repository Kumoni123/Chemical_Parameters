# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 22:18:55 2024

@author: HP
"""

import pandas as pd
import numpy as np

from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from datetime import *

column_mapping = ColumnMapping()
data_drift = Report(metrics=[DataDriftPreset()])


new_data = pd.read_csv('data/predictions.csv', index_col=[0],parse_dates=[0])
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


