import streamlit as st
from clustering import (
    cargar_datos,
    identificar_codo,
    metodo_del_codo,
    preprocesar,
    visualizar_pca,
    visualizar_tsne,
    visualizar_umap,
    generar_clusters_kmeans,
    visualizacion,
)

# Título y descripción de la aplicación
st.title("Análisis de Datos: German Credit Data")
st.markdown("""
Esta aplicación permite analizar y visualizar datos del conjunto *German Credit Data* utilizando diversas técnicas de análisis y clustering.
""")

# Cargar datos
st.sidebar.header("Opciones")
df = cargar_datos()
numerical_cols = ['Age', 'Credit amount', 'Duration']

st.write("Vista previa de los datos:")
st.dataframe(df.head())

# Selección de análisis
opcion = st.sidebar.selectbox(
    "Selecciona una acción:",
    [
        "Método del Codo",
        "Visualización PCA",
        "Visualización t-SNE",
        "Visualización UMAP",
        "Generar Clusters",
        "Visualización de Clusters",
    ],
)

# Procesar datos
processed_data = preprocesar(df)

# Métodos seleccionados
if opcion == "Método del Codo":
    st.subheader("Método del Codo para determinar k")
    fig = metodo_del_codo(df)
    st.pyplot(fig)

elif opcion == "Visualización PCA":
    st.subheader("Visualización PCA")
    fig = visualizar_pca(df, numerical_cols)
    st.pyplot(fig)

elif opcion == "Visualización t-SNE":
    st.subheader("Visualización t-SNE")
    fig = visualizar_tsne(processed_data)
    st.pyplot(fig)

elif opcion == "Visualización UMAP":
    st.subheader("Visualización UMAP")
    fig = visualizar_umap(processed_data)
    st.pyplot(fig)

elif opcion == "Generar Clusters":
    st.subheader("Generar Clusters con K-Means")
    k = st.sidebar.slider("Selecciona el número de clusters (k):", 2, 10, 4)
    generar_clusters_kmeans(df, k)
    st.success(f"Clusters generados con k={k} y guardados en 'german_credit_data_results.csv'.")

elif opcion == "Visualización de Clusters":
    st.subheader("Visualización de Clusters")
    fig, data_results = visualizacion()
    st.pyplot(fig)
    st.write("Datos con clusters:")
    st.dataframe(data_results.head())
