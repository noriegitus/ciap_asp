import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

# Cargar datos
def cargar_datos(ruta='datos.csv'):
    """
    Carga un dataset desde un archivo CSV y devuelve un DataFrame.
    """
    try:
        df = pd.read_csv(ruta)
        print("Datos cargados correctamente.")
        return df
    except FileNotFoundError:
        print(f"Error: El archivo '{ruta}' no fue encontrado.")
        raise

# Preprocesar datos
def preprocesar(df):
    """
    Preprocesa los datos eliminando columnas no numéricas y estandarizando las columnas numéricas.
    Devuelve los datos procesados y los nombres de las columnas numéricas.
    """
    # Selección de columnas numéricas
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

    if not numerical_cols:
        raise ValueError("El DataFrame no contiene columnas numéricas para procesar.")

    # Escalar datos
    scaler = StandardScaler()
    processed_data = scaler.fit_transform(df[numerical_cols])

    print("Datos preprocesados correctamente.")
    return processed_data, numerical_cols

# Método del Codo
def metodo_del_codo(processed_data):
    """
    Calcula las distorsiones para diferentes valores de k usando KMeans.
    Devuelve una lista de distorsiones.
    """
    distortions = []
    K = range(1, 11)
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(processed_data)
        distortions.append(kmeans.inertia_)

    print("Método del Codo ejecutado correctamente.")
    return distortions

# Identificar el codo
def identificar_codo(distortions):
    """
    Identifica el número óptimo de clusters basándose en la lista de distorsiones.
    """
    if len(distortions) < 3:
        raise ValueError("Se necesitan al menos 3 valores de distorsión para identificar el codo.")

    diferencias = [distortions[i] - distortions[i + 1] for i in range(len(distortions) - 1)]
    optimal_k = diferencias.index(max(diferencias)) + 1

    print(f"Número óptimo de clusters identificado: {optimal_k}")
    return optimal_k

# Generar clusters
def generar_clusters_kmeans(processed_data, k):
    """
    Genera clusters utilizando KMeans con el número de clusters especificado (k).
    Devuelve las etiquetas de los clusters.
    """
    if not isinstance(k, int) or k <= 0:
        raise ValueError("El número de clusters (k) debe ser un entero positivo.")

    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(processed_data)

    print("Clusters generados correctamente.")
    return kmeans.labels_

# Visualizar Método del Codo
def visualizar_metodo_del_codo(distortions):
    """
    Genera un gráfico del Método del Codo.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, 11), distortions, marker='o', linestyle='--')
    plt.xlabel('Número de Clusters')
    plt.ylabel('Distorsión')
    plt.title('Método del Codo')
    plt.grid()
    plt.show()

# Visualizar Pairplot
def visualizar_pairplot(df, numerical_cols, clusters):
    """
    Genera un pairplot con los clusters identificados.
    """
    df_clusters = df.copy()
    df_clusters['Cluster'] = clusters

    sns.pairplot(df_clusters, vars=numerical_cols, hue='Cluster', palette='tab10')
    plt.show()

# Flujo principal
if __name__ == "__main__":
    try:
        # Cargar los datos
        ruta_csv = 'datos.csv'  # Cambia esta ruta si el archivo no está en el directorio actual
        df = cargar_datos(ruta=ruta_csv)

        # Preprocesar datos
        processed_data, numerical_cols = preprocesar(df)

        # Método del Codo
        distortions = metodo_del_codo(processed_data)
        optimal_k = identificar_codo(distortions)

        # Visualizar Método del Codo
        visualizar_metodo_del_codo(distortions)

        # Generar clusters
        clusters = generar_clusters_kmeans(processed_data, k=optimal_k)

        # Visualizar clusters
        visualizar_pairplot(df, numerical_cols, clusters)

    except Exception as e:
        print(f"Error: {e}")
