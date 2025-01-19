import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# 1.1 Cargar datos
df = pd.read_csv("german_credit_data.csv", index_col=0)

# Observa las primeras filas y estructura de datos
print(df.head())
print(df.info())

# 1.2 Reemplazar valores 'NA' en columnas categóricas con 'Unknown'
df['Saving accounts'] = df['Saving accounts'].replace('NA', 'Unknown')
df['Checking account'] = df['Checking account'].replace('NA', 'Unknown')

# 1.3 Separar la columna 'Risk' para uso posterior en análisis
df_risk = df['Risk'].copy()   # guardamos la columna de riesgo
df_for_clustering = df.drop(columns=['Risk'])

# 2. Manejo de valores nulos

# 2.1 Imputar nulos en columnas categóricas: reemplazar NaN por 'Unknown'
categorical_cols = ['Sex', 'Housing', 'Saving accounts', 'Checking account', 'Purpose']
for col in categorical_cols:
    df_for_clustering[col] = df_for_clustering[col].fillna('Unknown')

# 2.2 Imputar nulos en columnas numéricas con la mediana
numeric_cols = df_for_clustering.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    median_value = df_for_clustering[col].median()
    df_for_clustering[col].fillna(median_value, inplace=True)

# 3. Codificar variables categóricas
label_encoder = LabelEncoder()
for col in categorical_cols:
    df_for_clustering[col] = label_encoder.fit_transform(df_for_clustering[col].astype(str))

# 4. Escalar los datos
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_for_clustering)

# 5. Seleccionar k usando método del codo y silhouette
inertias = []
sil_scores = []
K_range = range(2, 10)

for k in K_range:
    kmeans_temp = KMeans(n_clusters=k, random_state=42)
    labels_temp = kmeans_temp.fit_predict(df_scaled)
    
    inertias.append(kmeans_temp.inertia_)
    sil = silhouette_score(df_scaled, labels_temp)
    sil_scores.append(sil)

# Gráfica del codo y silhouette
plt.figure(figsize=(12,4))

plt.subplot(1,2,1)
plt.plot(K_range, inertias, 'o--')
plt.title("Método del codo")
plt.xlabel("Número de clusters (k)")
plt.ylabel("Inercia")

plt.subplot(1,2,2)
plt.plot(K_range, sil_scores, 'o--', color='green')
plt.title("Índice silhouette")
plt.xlabel("Número de clusters (k)")
plt.ylabel("Silhouette score")

plt.tight_layout()
plt.show()

# 6. Entrenar el modelo final con k=5
optimal_k = 5  # Actualizado a 5 clusters
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
cluster_labels = kmeans.fit_predict(df_scaled)

# Añadir la columna de cluster al DataFrame original
df['Cluster'] = cluster_labels

# 7. Análisis de clusters

# 7.1 Pivot table de Risk por Cluster
pivot_risk = df.pivot_table(index='Cluster', columns='Risk', aggfunc='size', fill_value=0)
print("Pivot table de cuentas según riesgo por cluster:")
print(pivot_risk)

# 7.2 Porcentajes de riesgo por cluster
total_by_cluster = df.groupby('Cluster')['Risk'].count()
pivot_risk_percent = pivot_risk.div(total_by_cluster, axis=0) * 100
print("\nPorcentaje de riesgo por cluster:")
print(pivot_risk_percent)

# 7.3 Análisis de características por cluster (numéricas)
cluster_analysis = df.groupby('Cluster').mean(numeric_only=True)
print("\nAnálisis de características promedio por cluster:")
print(cluster_analysis)

# 7.4 Distribución de variables categóricas por cluster
print("\nDistribución de 'Sex' por cluster (%):")
sex_distribution = df.groupby('Cluster')['Sex'].value_counts(normalize=True) * 100
print(sex_distribution)

print("\nDistribución de 'Housing' por cluster (%):")
housing_distribution = df.groupby('Cluster')['Housing'].value_counts(normalize=True) * 100
print(housing_distribution)

# Puedes repetir para otras variables categóricas como 'Saving accounts', 'Checking account', 'Purpose'

# 8. Visualización con PCA
pca = PCA(n_components=2)
pcs = pca.fit_transform(df_scaled)

plt.figure(figsize=(6,4))
sns.scatterplot(x=pcs[:,0], y=pcs[:,1], hue=cluster_labels, palette="Set2")
plt.title("Clusters en 2D con PCA (k=5)")
plt.xlabel("Primer Componente Principal (PC1)")
plt.ylabel("Segundo Componente Principal (PC2)")
plt.show()

# Imprimir componentes principales para entender loadings
print("\nVectores de carga de PCA:")
print(pca.components_)
