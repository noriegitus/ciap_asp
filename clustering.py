import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline

def cargar_datos():
    # Cargar los datos
    df = pd.read_csv('german_credit_data.csv')
    df['Risk_encoded'] = df['Risk'].map({'good': 0, 'bad': 1})
    return df

def visualizar_pca(df):
    numerical_cols = ['Age', 'Credit amount', 'Duration']
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[numerical_cols])

    # PCA para visualización inicial
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled_data)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=pca_result[:, 0], y=pca_result[:, 1], hue=df['Risk'], palette="coolwarm")
    plt.title('PCA de variables numéricas')
    return plt.gcf()

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
    return plt.gcf()

def visualizar_clusters(df, k=4):
    numerical_cols = ['Age', 'Credit amount', 'Duration']
    categorical_cols = ['Saving accounts', 'Checking account', 'Housing', 'Purpose']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(drop='first'), categorical_cols)
        ])
    processed_data = preprocessor.fit_transform(df)

    # Clustering
    kmeans = KMeans(n_clusters=k, random_state=42)
    df['Cluster'] = kmeans.fit_predict(processed_data)

    # PCA para visualización de clusters
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(processed_data)

    df_pca = pd.DataFrame(pca_result, columns=['PCA1', 'PCA2'])
    df_pca['Cluster'] = df['Cluster']

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', data=df_pca, palette='Set2', s=100)
    plt.title('Visualización de Clusters (PCA)')
    return plt.gcf(), df
