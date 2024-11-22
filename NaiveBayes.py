from extr_trans_carac import df_airlines

print("Pereme tantito, le ando echando ganas")

# Discretizar ARR_DELAY en categorías
def categorizar_retraso(delay):
    if delay >= 15:
        return 'Delay'
    elif delay <= -15:
        return 'Early'
    else:
        return 'OnTime'
    
df_airlines['ARR_DELAY_CATEGORY'] = df_airlines['ARR_DELAY'].apply(categorizar_retraso)

# Selección de muestra
sample_size = 0.2  
df_sample = df_airlines.sample(frac=sample_size, random_state=42)

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Escalar los datos (solo las columnas numéricas)
X_scaled = StandardScaler().fit_transform(df_sample.drop(columns=['ARR_DELAY', 'ARR_DELAY_CATEGORY']))

# Aplicar PCA para reducir la dimensionalidad
pca = PCA(n_components=100, random_state=42)
X_pca = pca.fit_transform(X_scaled)

# Definir X (características) e y (variable objetivo)
X = X_pca
y = df_sample['ARR_DELAY_CATEGORY']

# Paso 3: Dividir los datos en entrenamiento y prueba
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Paso 4: Entrenamiento del modelo Naive Bayes
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Entrenar el modelo
modelo_nb = GaussianNB()
modelo_nb.fit(X_train, y_train)

# Predicción y evaluación
y_pred = modelo_nb.predict(X_test)

# Reporte de resultados
print("Matriz de confusión:")
print(confusion_matrix(y_test, y_pred))

print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred))

print("\nExactitud del modelo:")
print(accuracy_score(y_test, y_pred))