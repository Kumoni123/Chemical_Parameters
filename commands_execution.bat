@echo off && echo Ejecutando comandos... && C:\ProgramData\anaconda3\Scripts\activate.bat && conda activate ML_PQ && cd .\Scripts && python mainPQ.py && conda deactivate && conda deactivate && cd.. && echo Comando completado.