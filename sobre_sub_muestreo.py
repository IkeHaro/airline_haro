from sklearn.utils import resample
import pandas as pd

# Separar las clases
df_majority = df_airlines[df_airlines['ARR_DELAY_CATEGORY'] == 'OnTime']
df_minority_early = df_airlines[df_airlines['ARR_DELAY_CATEGORY'] == 'Early']
df_minority_delay = df_airlines[df_airlines['ARR_DELAY_CATEGORY'] == 'Delay']


##SOBREMUESTREO##
# Aumentar las clases minoritarias al tamaño de la clase mayoritaria
majority_size = len(df_majority)
df_minority_early_upsampled = resample(df_minority_early,replace=True,n_samples=majority_size,random_state=42)
df_minority_delay_upsampled = resample(df_minority_delay,replace=True,n_samples=majority_size,random_state=42)

# Combinar las clases balanceadas
df_balanced = pd.concat([df_majority, df_minority_early_upsampled, df_minority_delay_upsampled])

# Reducir la clase mayoritaria al tamaño de la clase minoritaria más pequeña
minority_size = min(len(df_minority_early), len(df_minority_on_time))
df_majority_downsampled = resample(df_majority,replace=False,n_samples=minority_size,random_state=42)

# Combinar las clases balanceadas
df_balanced = pd.concat([df_majority_downsampled, df_minority_early, df_minority_on_time])
