import streamlit as st
# libreria 
import pandas as pd
import numpy as np

# titulos y divisores generales
st.title("Draft del Dashboard", False)
st.header("Header General",anchor=False,help="detallitos",divider="rainbow")

# escritura general
st.markdown("### Tablas / Dataframes basics")

# uso del feature magia para evitar escribir st.write()
st.write("Ejemplo de `st.write()` -> tablas interactivas")
data_frame = pd.DataFrame({
    'columna 1': [1,"lolazo",3,"valorant god"],
    'columna 2': ["perros","gatos","murcielagos","mapaches"]
})
data_frame

# uso de st.table en vez de st.write
st.write("Ejemplo de `st.table()` -> tablas estaticas")
st.table(data_frame)

# manejo de index con st.table
st.markdown("-> Uso de `st.table(df.set_index('columna 1'))`")
st.table(data_frame.set_index('columna 1'))

st.markdown("-> Uso de `st.table(df.set_index('columna 1'))`")
st.table(data_frame.values)

# manejo de index con st.dataframe, mas recomendable (tabla interactiva default)
st.write("Ejemplo de `st.dataframe(df, hide_index=True)`")
st.dataframe(data_frame, hide_index=True)

# division del espacio
st.markdown("### Division del espacio")
st.write("Ejemplo de `st.columns(int)`")

# dataframe 1
df1 = pd.DataFrame({
    'animal': ["perro","gato","murcielago","mapache"],
    'nombres': ["bianka","biankita","isaac","jeremi"]
})

#dataframe 2
df2 = pd.DataFrame({
    'edades': [1312,2,3,4],
    'numeros': [3,435,323,123]
})

df3 = pd.DataFrame({
    'dias': [1,2,3,4],
    'luces': [23,51,5,20]
})

# BLOQUE DE COLUMNAS 1 (contiene 2 columnas)
col1, col2  = st.columns(2)

# tablas base demo
with col1:
    st.write("### Visual 1")
    st.dataframe(df1, hide_index=True)

with col2:
    st.write("### Visual 2")
    st.dataframe(df2,hide_index=True)
    
# BLOQUE DE COLUMNAS 2 (contiene 3 columnas)
col3, col4, col5 = st.columns(3)
    
# estilo de tablas basico
st.write("-> usando el objeto `styler` de pandas ")
with col3:
    st.write("### Visual 1")
    st.write("`df.style.highlight_max(axis=0)` para resaltar el maximo valor por COLUMNA")
    # highlight_max(solo para columnas numericas)
    st.dataframe(df1.style.highlight_max(axis=0), hide_index=True)

with col4:
    st.write("### Visual 2")
    st.write("`df.style.highlight_max(axis=1)` para resaltar el maximo valor por FILA")
    st.dataframe(df2.style.highlight_max(axis=1), hide_index=True)

with col5:
    st.write("### Visual 3")
    st.write("`df.style.highlight_max(axis=None)` para resaltar el maximo valor de la TAblA")
    st.dataframe(df3.style.highlight_max(axis=None), hide_index=True)
    

# Line Chart 

st.markdown("## Line Charts, Diagramas de Linea")
st.markdown("Se creo con pandas un data frame y se lo relleno de un arreglo con numpy `np.random.randn(20, 3)`")

chart_data = pd.DataFrame(
     np.random.randn(20, 3),  # genera una matriz (arreglo) aleatoria de 20 filas y 3 columnas
     columns=['a', 'b', 'c'])  # nombra las columnas como a b c
st.line_chart(chart_data)

st.markdown("## Plot Charts, Grafico de Trazado/Coordenadas")
st.markdown("Se creo con pandas un data frame, y se lo relleno de un arreglo con numpy `np.random.rand(1000, 2) / [50, 500] + [37.76, -122.4]`")
map_data = pd.DataFrame(
    np.random.randn(1000,2)/[50,500]+[37.76, -122.4], # matriz de 1000 filas y 2 columnas con distribucion normal. puntos rand en 2d
    columns = ['lat', 'lon'] # tiene que si o si llamarse columns, igual con lat y lon pq asi streamlit los interpreta como datos geograficos
)  
st.map(map_data)
st.markdown("### datazos:")
st.markdown("`(1000,2)` Genera una matriz de datos rand en 2D que se usaran para modelar ubs geo`")  
st.markdown("`/[50,50]` Div las columns latitut y longuitud por 50 y 500 para reducir el rango de variacion de cada punto. Permitiendo una agrupacion mas cercana al centro")
st.markdown("`+[37.76, -122.4]` Es un desplazamiento base (punto central)")
st.markdown("37.76 Es una latitud base (San Francisco, Cal)")
st.markdown("-122.4 Es una longitud base (Tambien perteneciente a SF, Cal)")
st.markdown("Cada valor generado se desplazara alrededor de estos puntos centrales")
