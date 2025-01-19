import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline

# Configuración de estilo para las gráficas
sns.set(style="whitegrid")

# Función para cargar y procesar los datos
@st.cache_data
def cargar_datos():
    df = pd.read_csv('german_credit_data.csv')
    df['Risk_encoded'] = df['Risk'].map({'good': 0, 'bad': 1})
    return df

# Función para visualizar PCA inicial
def visualizar_pca(df):
    st.subheader('Análisis de Componentes Principales (PCA)')
    st.write('El PCA reduce la dimensionalidad de los datos numéricos, permitiendo visualizar su estructura en 2D. A continuación, se muestra la distribución de los datos según el riesgo crediticio.')

    numerical_cols = ['Age', 'Credit amount', 'Duration']
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[numerical_cols])

    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled_data)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x=pca_result[:, 0], y=pca_result[:, 1], hue=df['Risk'], palette="coolwarm", ax=ax)
    ax.set_title('PCA de Variables Numéricas')
    ax.set_xlabel('Componente Principal 1')
    ax.set_ylabel('Componente Principal 2')
    st.pyplot(fig)
    st.write('En esta gráfica, cada punto representa un cliente, y los colores indican el riesgo asociado. La superposición de colores sugiere que las variables numéricas por sí solas pueden no ser suficientes para separar claramente los grupos de riesgo.')

# Función para determinar el número óptimo de clusters usando el método del codo
def metodo_del_codo(df):
    st.subheader('Determinación del Número Óptimo de Clusters: Método del Codo')
    st.write('El método del codo ayuda a identificar el número óptimo de clusters al observar la disminución de la inercia (suma de distancias al cuadrado dentro de los clusters) a medida que aumenta k.')

    numerical_cols = ['Age', 'Credit amount', 'Duration']
    categorical_cols = ['Saving accounts', 'Checking account', 'Housing', 'Purpose']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(drop='first'), categorical_cols)
        ])
    processed_data = preprocessor.fit_transform(df)

    inertia = []
    k_range = range(2, 11)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(processed_data)
        inertia.append(kmeans.inertia_)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(k_range, inertia, marker='o')
    ax.set_title('Método del Codo para Determinar k')
    ax.set_xlabel('Número de Clusters (k)')
    ax.set_ylabel('Inercia')
    st.pyplot(fig)
    st.write('Busca el "codo" en la gráfica, donde la disminución de la inercia se vuelve menos pronunciada. Este punto sugiere el número óptimo de clusters. Por ejemplo, si el codo se encuentra en k=4, este sería el número recomendado de clusters.')

# Función para realizar clustering y visualizar los resultados
def visualizar_clusters(df, k=4):
    st.subheader(f'Visualización de Clusters con k={k}')
    st.write('Se aplica K-Means para segmentar a los clientes en grupos similares. A continuación, se muestra la distribución de los clusters en el espacio reducido por PCA.')

    numerical_cols = ['Age', 'Credit amount', 'Duration']
    categorical_cols = ['Saving accounts', 'Checking account', 'Housing', 'Purpose']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(drop='first'), categorical_cols)
        ])
    processed_data = preprocessor.fit_transform(df)

    kmeans = KMeans(n_clusters=k, random_state=42)
    df['Cluster'] = kmeans.fit_predict(processed_data)

    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(processed_data)

    df_pca = pd.DataFrame(pca_result, columns=['PCA1', 'PCA2'])
    df_pca['Cluster'] = df['Cluster']

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', data=df_pca, palette='Set2', s=100, ax=ax)
    ax.set_title('Visualización de Clusters (PCA)')
    ax.set_xlabel('Componente Principal 1')
    ax.set_ylabel('Componente Principal 2')
    st.pyplot(fig)
    st.write('Cada punto representa un cliente, y los colores indican el cluster asignado. La separación de los clusters en el espacio PCA sugiere que el modelo ha identificado grupos distintos de clientes con características similares.')

    return df

# Función para mostrar el resumen de los clusters
def resumen_clusters(df):
    st.subheader('Resumen de los Clusters')
    st.write('A continuación, se presenta un resumen de las características promedio de cada cluster, lo que permite entender las diferencias entre los grupos de clientes.')

    cluster_summary = df.groupby('Cluster').mean(numeric_only=True)
    st.dataframe(cluster_summary)
    st.write('Este resumen muestra las medias de las variables numéricas para cada cluster, proporcionando una visión general de las características de cada grupo de clientes.')

# Aplicación principal de Streamlit
def main():
    st.title('Análisis de Clustering de Clientes')
    st.write('Esta aplicación permite explorar los resultados del análisis de clustering aplicado a los datos de crédito alemán.')

    df = cargar_datos()

    st.sidebar.header('Parámetros de Clustering')
    if st.sidebar.checkbox('Mostrar Análisis de PCA Inicial'):
        visualizar_pca(df)

    if st.sidebar.checkbox('Mostrar Método del Codo para Determinar k'):
        metodo_del_codo(df)

    k = st.sidebar.slider('Selecciona el número de clusters (k)', min_value=2, max_value=10, value=4)
    if st.sidebar.button('Aplicar Clustering'):
        df = visualizar_clusters(df, k)
        resumen_clusters(df)

if __name__ == "__main__":
    main()
