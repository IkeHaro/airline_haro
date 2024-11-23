def dist_variables_numericas():
    # Distribución de las variables numéricas
    columnas_numericas = df.select_dtypes(include=['float64', 'int64']).columns
    df[columnas_numericas].hist(bins=15, figsize=(15, 10))
    plt.suptitle("Distribución de las variables numéricas")
    plt.show()
    
def boxplot_columnas(columnas):

    plt.figure(figsize=(15, 10))
    for i, col in enumerate(columnas):
        plt.subplot(1, len(columnas), i + 1)  # Crear subplots en una sola fila
        sns.boxplot(y=df[col], color='skyblue')
        plt.title(f"Boxplot de {col}")
        plt.xlabel(col)
    plt.tight_layout()
    plt.show()

columnas_seleccionadas = ['DEP_TIME', 'DEP_DELAY', 'TAXI_OUT', 'WHEELS_OFF', 'ARR_DELAY']
boxplot_columnas(columnas_seleccionadas)
dist_variables_numericas()