# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 23:29:08 2024

@author: HP
"""

import logging
import logging.config  # Por alguna razon esto esta en otro modulo...
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PowerTransformer
import pickle

logger = logging.getLogger("my_final_project")  # Common is to use __name__ here

os.makedirs("logs", exist_ok=True)

# siiii un diccionario
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,  # This is important
    "formatters": {
        "simple": {
            "format": "[%(levelname)s|%(module)s|%(lineno)d] %(asctime)s: %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S%z",  # Use iso8601 and include timezone
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",  # Si usas los handlers por defecto tienes que usar "class"
            "level": "ERROR",
            "formatter": "simple",
            "stream": "ext://sys.stdout",  # This is the default value for stream to stdout
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",  # Handler para un archivo que rota. (Por alguna razon esta escondido tambien)
            "level": "DEBUG",
            "formatter": "simple",
            "filename": "logs/dict.log",
            "maxBytes": 2048,  # Escribe 2KB y el backup (Rota)
            "backupCount": 3,  # Guarda 3 archivos de backup
        },
    },
    "loggers": {
        "root": {
            "level": "DEBUG",
            "handlers": ["console", "file"],
        }
    },
}


# Basic logger configuration this goes usually in a separate file config/logging.py
def logger_config():
    logging.config.dictConfig(LOGGING_CONFIG)


def main():
    logger_config()
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")
    try:
        df_new = pd.read_csv('data/new_data.csv', index_col=[0],parse_dates=[0])
    except:
        logger.exception("No se encontró la data")  # ERROR LEVEL
        
    
    try:
        sc = MinMaxScaler(feature_range=(0,1))
        xarray = sc.fit_transform(np.array(df_new))
        
        pt = PowerTransformer(method = 'yeo-johnson', standardize=True)
        skl_boxcox = pt.fit(xarray) #1ra columna
        
        xarray = pt.transform(xarray)
        
        df_out = pd.DataFrame(xarray, columns=['Kc', 'FW_MUESTRA'])
    except:
        logger.exception("Fallo en el preprocesamiento")  # ERROR LEVEL

    try:
        
        # model_knn = pickle.load(open('./model/modelVC.pkl', 'rb'))
        model_kc = pickle.load(open('model/model.pkl', 'rb'))
        
        y_pred = model_kc.predict(xarray)
        
        df_new['Predicción'] =  y_pred
        df_new.to_csv('output/predictions.csv')
    except:
        logger.exception("Fallo en la predicción")  # ERROR LEVEL


if __name__ == "__main__":
    main()