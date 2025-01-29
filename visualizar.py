import streamlit as st
from clustering import (
    cargar_datos,
    preprocesar,
    metodo_del_codo,
    determinar_k_optimo,
    generar_clusters_kmeans,
    visualizar_metodo_del_codo,
    visualizar_pairplot
)

# Configuración inicial de Streamlit
st.set_page_config(page_title="Análisis de Clustering", layout="wide")

# Título principal
st.title("Análisis de Clustering de Datos")

# Paso 1: Cargar los datos
st.header("1. Cargar Datos")
try:
    df = cargar_datos()
    st.write("Datos cargados con éxito. Vista previa del dataset:")
    st.dataframe(df.head())
except FileNotFoundError:
    st.error("Error: No se encontró el archivo de datos. Asegúrate de que el archivo 'datos.csv' esté en el directorio correcto.")
    st.stop()

# Paso 2: Preprocesamiento
st.header("2. Preprocesamiento de Datos")
processed_data, numerical_cols = preprocesar(df)
st.write("Datos preprocesados exitosamente.")
st.write("Columnas numéricas detectadas y escaladas:")
st.write(numerical_cols)

# Paso 3: Método del Codo
st.header("3. Determinación del Número Óptimo de Clusters")
if st.button("Ejecutar Método del Codo"):
    distortions = metodo_del_codo(processed_data)
    optimal_k = determinar_k_optimo(distortions)
    st.write(f"Número óptimo de clusters identificado: {optimal_k}")
    
    # Visualizar el método del codo
    fig_codo = visualizar_metodo_del_codo(distortions)
    st.pyplot(fig_codo)

# Paso 4: Generar Clusters
st.header("4. Generar Clusters")
if st.button("Generar Clustering"):
    if 'optimal_k' in locals():
        clusters = generar_clusters_kmeans(processed_data, k=optimal_k)
        df['Cluster'] = clusters  # Agregar los clusters al DataFrame original
        st.write("Clustering generado con éxito. Vista previa del dataset con clusters:")
        st.dataframe(df.head())
    else:
        st.error("Primero debes ejecutar el método del codo para determinar el número óptimo de clusters.")

# Paso 5: Visualización con Pairplot
st.header("5. Visualización de Clusters con Pairplot")
if st.button("Mostrar Pairplot"):
    if 'clusters' in locals():
        st.write("Generando visualización de clusters con pairplot...")
        visualizar_pairplot(df, numerical_cols, clusters)
    else:
        st.error("Primero debes generar los clusters para visualizar el pairplot.")
