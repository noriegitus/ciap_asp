import streamlit as st
# libreria 
import pandas as pd

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

# asignar columas
col1, col2, col3 = st.columns(3)

# tablas base demo
with col1:
    st.write("## Visual 1")
    st.dataframe(df1, hide_index=True)

with col2:
    st.write("## Visual 2")
    st.dataframe(df2,hide_index=True)
    
# estilo de tablas basico
st.write("-> usando el objeto `styler` de pandas ")
with col1:
    st.write("## Visual 1")
    st.write("`df.style.highlight_max(axis=0)` para resaltar el maximo valor por COLUMNA")
    # highlight_max(solo para columnas numericas)
    st.dataframe(df1.style.highlight_max(axis=0), hide_index=True)

with col2:
    st.write("## Visual 2")
    st.write("`df.style.highlight_max(axis=1)` para resaltar el maximo valor por FILA")
    st.dataframe(df2.style.highlight_max(axis=1), hide_index=True)

with col3:
    st.write("## Visual 3")
    st.write("`df.style.highlight_max(axis=None)` para resaltar el maximo valor de la TAblA")
    st.dataframe(df3.style.highlight_max(axis=None), hide_index=True)