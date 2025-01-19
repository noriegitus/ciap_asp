import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
import umap.umap_ as umap
from sklearn.manifold import TSNE
from kneed import KneeLocator


def cargar_datos():
    # Cargar los datos
    df = pd.read_csv('german_credit_data.csv')
    df['Risk_encoded'] = df['Risk'].map({'good': 0, 'bad': 1})
    return df


def identificar_codo(inertia):
    # Identifica el valor óptimo de k utilizando la biblioteca kneed. 
    k_range = range(2, len(inertia) + 2)  # Los valores de k correspondientes a las inercias
    kn = KneeLocator(k_range, inertia, curve="convex", direction="decreasing")
    return kn.knee


def metodo_del_codo(df):
    numerical_cols = ['Age', 'Credit amount', 'Duration']
    categorical_cols = ['Saving accounts', 'Checking account', 'Housing', 'Purpose']

    # Preprocesamiento
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(drop='first'), categorical_cols)
        ])
    processed_data = preprocessor.fit_transform(df)

    # Método del Codo
    inertia = []
    k_range = range(2, 10)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(processed_data)
        inertia.append(kmeans.inertia_)

    plt.figure(figsize=(8, 4))
    plt.plot(k_range, inertia, marker='o')
    plt.title('Método del Codo para determinar k')
    plt.xlabel('Número de Clusters (k)')
    plt.ylabel('Inercia')
    # Identificar el codo y resaltarlo
    optimal_k = identificar_codo(inertia)  # Implementa esta función según tu criterio
    plt.axvline(x=optimal_k, color='red', linestyle='--', label=f'k óptimo = {optimal_k}')
    plt.legend()
    return plt.gcf()

def preprocesar(df): 
    numerical_cols = ['Age', 'Credit amount', 'Duration']
    categorical_cols = ['Saving accounts', 'Checking account', 'Housing', 'Purpose']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(drop='first'), categorical_cols)
        ])
    processed_data = preprocessor.fit_transform(df)
    return processed_data

def visualizar_pca(processed_data, numerical_cols):
# Escalar los datos numéricos originales
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(processed_data[numerical_cols])

    # PCA para reducir a 2 dimensiones
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled_data)

    # Visualización con seaborn
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        x=pca_result[:, 0], 
        y=pca_result[:, 1], 
        hue=processed_data['Risk'], 
        palette="coolwarm"
    )
    plt.title('PCA de variables numéricas')
    return plt.gcf()

def visualizar_tsne(processed_data):
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    tsne_result = tsne.fit_transform(processed_data)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=tsne_result[:, 0], y=tsne_result[:, 1], palette="coolwarm")
    plt.title('t-SNE de datos preprocesados')
    return plt.gcf()

def visualizar_umap(processed_data):
    reducer = umap.UMAP(n_components=2, random_state=42)
    umap_result = reducer.fit_transform(processed_data)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=umap_result[:, 0], y=umap_result[:, 1], palette="coolwarm")
    plt.title('UMAP de datos preprocesados')
    return plt.gcf()

def generar_clusters_kmeans(df, k=4):
    # Clustering
    kmeans = KMeans(n_clusters=k, random_state=42)
    df['Cluster'] = kmeans.fit_predict(processed_data)
    df.to_csv("german_credit_data_results.csv") # guardar

def visualizacion():
    data_results = pd.read_csv("german_credit_data_results.csv")
    # PCA para visualización de clusters
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(processed_data)

    df_pca = pd.DataFrame(pca_result, columns=['PCA1', 'PCA2'])
    df_pca['Cluster'] = data_results['Cluster']

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', data=df_pca, palette='Set2', s=100)
    plt.title('Visualización de Clusters (PCA)')
    return plt.gcf(), data_results

