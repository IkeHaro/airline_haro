from limpieza_de_datos import eliminar_outliers
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('/Users/danny/OneDrive/Escritorio/airline_haro/train_airlines_delay_challenge.csv')
#df = pd.read_csv('/Users/danny/OneDrive/Documents/GitHub/airline_haro/train_airlines_delay_challenge.csv')

# Crear intervalos de tiempo para horas de salida y llegada
def categorizar_hora(hora):
    if 5 <= hora < 12:
        return 'mañana'
    elif 12 <= hora < 18:
        return 'tarde'
    elif 18 <= hora < 24:
        return 'noche'
    else:
        return 'madrugada'

df['DEP_TIME_CATEGORY'] = ((df['CRS_DEP_TIME'] // 100) % 24).apply(categorizar_hora)
df['ARR_TIME_CATEGORY'] = ((df['CRS_ARR_TIME'] // 100) % 24).apply(categorizar_hora)

# Extraer día de la semana y mes de la fecha
df['DAY_OF_WEEK'] = pd.to_datetime(df['FL_DATE']).dt.dayofweek
df['MONTH'] = pd.to_datetime(df['FL_DATE']).dt.month

# Calcular la velocidad media planeada
df['AVG_SPEED_PLANNED'] = df['DISTANCE'] / (df['CRS_ELAPSED_TIME'] / 60)

# Crear variable binaria para indicar si el retraso fue significativo
df['LATE_ARRIVAL'] = (df['ARR_DELAY'] > 15).astype(int)

# Convertir variables categóricas en dummies
df_airlines = pd.get_dummies(df, columns=['OP_CARRIER', 'ORIGIN', 'DEST', 'DEP_TIME_CATEGORY', 'ARR_TIME_CATEGORY'], drop_first=True)

# Eliminar columnas irrelevantes o ya procesadas
df_airlines = df_airlines.drop(columns=['FL_DATE', 'WHEELS_OFF', 'DEP_TIME', 'OP_CARRIER_FL_NUM'])

#print("Transformaciones completadas y nuevas características:")
#print(df_airlines.head())
