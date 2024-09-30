# Pasos para ejecutar correctamente los archivos
## 1. MLflow

Descomentar el modelo KNN y comentar el modelo LogisticRegression y viceversa para poder visualizar el versionamiento de los modelos.

## 2. Airflow

Instalar los requerimientos antes de ejecutar levantar el docker de Airflow.

## 3. Archivo .bat

El archivo commands_execution.bat es para el task scheduler de Windows que ejecuta el archivo cada 5 minutos.

## 4. Archivo environment.yml

Dicho archivo solo muestra la librería del PI Connect necesario para la puesta en producción, sin embargo, en el código se eliminó dicha parte porque se requeriría permisos especiales pero se dejó ahi dicho archivo para que se pueda visualizar las librerías utilizadas en la puesta en producción.
