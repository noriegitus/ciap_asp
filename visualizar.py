import streamlit as st
from clustering import cargar_datos, visualizar_pca, metodo_del_codo, visualizar_clusters

st.title("Análisis de Clustering con Streamlit")

# Cargar datos
st.subheader("Datos de entrada")
df = cargar_datos()
st.write(df.head())

# Visualización inicial (PCA)
st.subheader("Visualización inicial con PCA")
pca_plot = visualizar_pca(df)
st.pyplot(pca_plot)

# Método del Codo
st.subheader("Método del Codo para determinar el número óptimo de clusters")
codo_plot = metodo_del_codo(df)
st.pyplot(codo_plot)

# Selección de número de clusters
k = st.slider("Selecciona el número de clusters", 2, 10, 4)

# Visualización de clusters
st.subheader("Visualización de clusters")
clusters_plot, df_clustered = visualizar_clusters(df, k)
st.pyplot(clusters_plot)

# Resumen de clusters
st.subheader("Resumen por Cluster")
st.dataframe(df_clustered.groupby('Cluster').mean(numeric_only=True))
