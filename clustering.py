import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import umap.umap_ as umap

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def cargar_datos(ruta='datos.csv'):
    """
    Carga un dataset desde un archivo CSV y devuelve un DataFrame.
    """
    try:
        df = pd.read_csv(ruta, index_col=0)
        print("Datos cargados correctamente desde:", ruta)
        return df
    except FileNotFoundError:
        print(f"Error: El archivo '{ruta}' no existe.")
        raise
    except Exception as e:
        print(f"Error inesperado al cargar el archivo: {str(e)}")
        raise

def eliminar_outliers(df, numeric_cols):
    """
    Elimina outliers de las columnas numéricas especificadas
    usando el método IQR (Rango intercuartílico).
    
    Para cada columna numérica, calcula Q1 y Q3, determina el IQR
    (Q3 - Q1) y define límites inferior y superior:
    - Límite inferior = Q1 - 1.5 * IQR
    - Límite superior = Q3 + 1.5 * IQR

    Se conservan únicamente las filas cuyos valores se encuentran
    dentro de [límite_inferior, límite_superior].
    """
    df_clean = df.copy()
    for col in numeric_cols:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        limite_inferior = Q1 - 1.5 * IQR
        limite_superior = Q3 + 1.5 * IQR

        # Filtrar filas que están dentro de los límites
        before_rows = df_clean.shape[0]
        df_clean = df_clean[(df_clean[col] >= limite_inferior) & (df_clean[col] <= limite_superior)]
        after_rows = df_clean.shape[0]
        print(f"Columna '{col}': Eliminados {before_rows - after_rows} outliers.")
    
    # Mostramos cuántos registros quedan tras la eliminación de outliers
    print(f"\nTras eliminar outliers: {df_clean.shape[0]} filas (de {df.shape[0]} originalmente).")
    
    return df_clean

def preprocesar(df):
    """
    Preprocesa los datos:
    1. Imputa valores nulos en 'Saving accounts' y 'Checking account' con la moda.
    2. Codifica de manera ordinal las columnas 'Saving accounts' y 'Checking account'.
    3. Codifica 'Sex' y 'Risk' de forma binaria (0 o 1).
    4. Aplica codificación one-hot a las variables categóricas 'Housing' y 'Purpose'.
    5. Estandariza (escalado) las columnas numéricas: 'Age', 'Credit amount', 'Duration'.
    """
    # 1. Imputar valores nulos con la moda
    mode_saving = df["Saving accounts"].mode()[0]
    mode_checking = df["Checking account"].mode()[0]
    df["Saving accounts"] = df["Saving accounts"].fillna(mode_saving)
    df["Checking account"] = df["Checking account"].fillna(mode_checking)

    # 2. Codificación ordinal
    ordinal_mapping = {
        "little": 0,
        "moderate": 1,
        "quite rich": 2,
        "rich": 3
    }
    df["Saving accounts"] = df["Saving accounts"].map(ordinal_mapping)
    df["Checking account"] = df["Checking account"].map(ordinal_mapping)

    # 3. Codificación binaria
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df['Risk'] = df['Risk'].map({'bad': 0, 'good': 1})

    # 4. Codificación one-hot
    df = pd.get_dummies(df, columns=['Housing'], prefix='Housing')
    df = pd.get_dummies(df, columns=['Purpose'], prefix='Purpose')

    # 5. Estandarización
    numeric_cols = ["Age", "Credit amount", "Duration"]
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    return df

def encontrar_numero_optimo_clusters_kmeans(X, max_clusters=10):
    """
    Aplica el método del codo (Elbow Method) para K-Means y
    retorna la inercia (SSE) para cada k en el rango [2, max_clusters].
    Además, grafica la inercia vs. el número de clústeres.
    """
    inercias = []
    k_values = range(2, max_clusters + 1)

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        inercias.append(kmeans.inertia_)

    # Graficar la inercia vs. k
    plt.figure(figsize=(6,4))
    plt.plot(k_values, inercias, marker='o')
    plt.title('Elbow Method para K-Means (espacio reducido)')
    plt.xlabel('Número de clústeres (k)')
    plt.ylabel('Inercia (SSE)')
    plt.xticks(k_values)
    plt.grid(True)
    plt.show()

    return inercias

if __name__ == "__main__":
    try:
        # -------------------------------------------------
        # 1. Cargar datos (ajusta la ruta a tu archivo CSV)
        # -------------------------------------------------
        ruta_csv = 'german_credit_data.csv'
        df = cargar_datos(ruta=ruta_csv)

        # -------------------------------------------------
        # 2. Eliminar outliers en columnas numéricas antes del preprocesamiento
        # -------------------------------------------------
        numeric_cols = ["Age", "Credit amount", "Duration"]
        df_sin_outliers = eliminar_outliers(df, numeric_cols)

        # -------------------------------------------------
        # 3. Preprocesar datos
        # -------------------------------------------------
        df_preprocesado = preprocesar(df_sin_outliers)

        # Convertimos el DataFrame preprocesado a array NumPy
        X = df_preprocesado.values.astype(np.float32)

        # -------------------------------------------------
        # 4. Reducción de Dimensionalidad con UMAP
        # -------------------------------------------------
        reducer = umap.UMAP(
            n_neighbors=30,   # Ajusta según el tamaño de tu dataset
            n_components=2,   # Queremos 2 dimensiones (por ejemplo)
            min_dist=0.1,     # Distancia mínima
            random_state=42
        )

        # Ajustamos UMAP en los datos completos
        X_umap = reducer.fit_transform(X)

        # Creamos un DataFrame para facilitar la visualización
        df_umap = pd.DataFrame(X_umap, columns=['dim_1', 'dim_2'])

        # -------------------------------------------------
        # 5. Determinar k óptimo en el espacio UMAP
        # -------------------------------------------------
        # inercias = encontrar_numero_optimo_clusters_kmeans(X_umap, max_clusters=10)
        # Observa la gráfica del codo y elige k.
        k_optimo = 6  # Por ejemplo

        # -------------------------------------------------
        # 6. Clustering K-Means sobre el espacio reducido
        # -------------------------------------------------
        kmeans = KMeans(n_clusters=k_optimo, random_state=42)
        clusters = kmeans.fit_predict(X_umap)

        # Verificamos que la longitud de 'clusters' coincide con df_preprocesado
        print(f"Longitud de 'clusters': {len(clusters)}")
        print(f"Longitud de 'df_preprocesado': {df_preprocesado.shape[0]}")

        # Agregamos la asignación de clúster al DataFrame de UMAP
        df_umap['cluster'] = clusters
        df_sin_outliers['cluster'] = clusters
        # Especifica la ruta y el nombre del archivo donde deseas guardar los datos
        ruta_archivo = 'df_sin_outliers_con_clusters.csv'

        # Guardar el DataFrame en un archivo CSV
        df_sin_outliers.to_csv(ruta_archivo, index=False)

        print(f"El DataFrame ha sido guardado exitosamente en {ruta_archivo}")
        print(df_sin_outliers.head())
        # Agregamos la asignación de clúster al DataFrame preprocesado
        df_preprocesado = df_preprocesado.copy()  # Aseguramos que estamos trabajando con una copia
        df_preprocesado['cluster'] = clusters

        # -------------------------------------------------
        # 7. Analizar resultados
        # -------------------------------------------------
        # Calcular porcentaje de clientes "bad risk" por clúster
        risk_dist = df_sin_outliers.groupby('cluster')['Risk'].value_counts(normalize=True).unstack().fillna(0)
        risk_dist['% Bad Risk'] = (risk_dist[0] * 100).round(1)
        print(risk_dist[['% Bad Risk']])
        # Analizar variables numéricas 
        numeric_stats = df_sin_outliers.groupby('cluster')[["Age", "Credit amount", "Duration"]].mean().round(1)
        print(numeric_stats)
        # Agrupar columnas one-hot de "Purpose"
        purpose_cols = [c for c in df_preprocesado.columns if c.startswith('Purpose_')]
        purpose_dist = df_preprocesado.groupby('cluster')[purpose_cols].mean().round(2)
        print(purpose_dist * 100)
        account_stats = df_sin_outliers.groupby('cluster')[["Saving accounts", "Checking account"]].mean().round(1)
        print(account_stats)
        # -------------------------------------------------
        # 8. Visualizar clusters en el espacio UMAP
        # -------------------------------------------------
        plt.figure(figsize=(6, 5))
        sns.scatterplot(
            data=df_umap,
            x='dim_1', 
            y='dim_2', 
            hue='cluster', 
            palette='tab10', 
            alpha=0.7
        )
        plt.title("Clustering K-Means sobre espacio UMAP (2D)")
        plt.xlabel("dim_1")
        plt.ylabel("dim_2")
        plt.legend(title='Clúster')
        plt.show()

    except Exception as e:
        print(f"Error durante la ejecución principal: {e}")
