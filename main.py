import pandas as pd
import statsmodels.api as sm
from factor_analyzer import FactorAnalyzer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Especifica la ruta de tu archivo Excel
archivo_excel = 'FMCC Actividad 4 nuevo.xlsx'
archivo_excel_analisis_cluster = 'FMCC Actividad 4 nuevo.xlsx'

# Carga el archivo Excel en un DataFrame de pandas
df = pd.read_excel(archivo_excel)
df1 = pd.read_excel(archivo_excel_analisis_cluster)


# Accede a las columnas del DataFrame
columnas = df.columns

# Imprime las columnas
print("Columnas en el archivo Excel:")
for columna in columnas:
    print(columna)




#ANALISIS DE FRECUENCIA
total = df['Total'].value_counts()
relacion_calidad_precio = df['Relación calidad-precio'].value_counts()
ubicacion = df['Ubicación'].value_counts()
servicio = df['Servicio'].value_counts()
habitaciones = df['Habitaciones'].value_counts()
limpieza = df['Limpieza'].value_counts()
calidad_sueño = df['Calidad del sueño'].value_counts()
# Imprime las frecuencias
print("Análisis de Frecuencias:")
print(total)
print(relacion_calidad_precio)
print(ubicacion)
print(servicio)
print(habitaciones)
print(limpieza)
print(calidad_sueño)

plt.figure(figsize=(10, 6))  # Ajusta el tamaño del gráfico según tus preferencias
servicio.plot(kind='bar', color='grey')

# Personaliza el gráfico
plt.title('Diagrama de Barras de la Columna "Total"')
plt.xlabel('Categorías')
plt.ylabel('Frecuencia')
plt.xticks(rotation=45, ha='right')  # Ajusta la rotación de las etiquetas en el eje x si es necesario
plt.grid(axis='y', linestyle='--', alpha=0.7)  # Añade una cuadrícula horizontal
plt.show()


plt.figure(figsize=(10, 6))  # Ajusta el tamaño del gráfico según tus preferencias
servicio.plot(kind='bar', color='green')

# Personaliza el gráfico
plt.title('Diagrama de Barras de la Columna "Relación calidad-precio"')
plt.xlabel('Categorías')
plt.ylabel('Frecuencia')
plt.xticks(rotation=45, ha='right')  # Ajusta la rotación de las etiquetas en el eje x si es necesario
plt.grid(axis='y', linestyle='--', alpha=0.7)  # Añade una cuadrícula horizontal
plt.show()


# Crear el diagrama de barras
plt.figure(figsize=(10, 6))  # Ajusta el tamaño del gráfico según tus preferencias
servicio.plot(kind='bar', color='yellow')

# Personaliza el gráfico
plt.title('Diagrama de Barras de la Columna "Ubicación"')
plt.xlabel('Categorías')
plt.ylabel('Frecuencia')
plt.xticks(rotation=45, ha='right')  # Ajusta la rotación de las etiquetas en el eje x si es necesario
plt.grid(axis='y', linestyle='--', alpha=0.7)  # Añade una cuadrícula horizontal
plt.show()

# Crear el diagrama de barras
plt.figure(figsize=(10, 6))  # Ajusta el tamaño del gráfico según tus preferencias
servicio.plot(kind='bar', color='blue')
# Personaliza el gráfico
plt.title('Diagrama de Barras de la Columna "Servicio"')
plt.xlabel('Categorías')
plt.ylabel('Frecuencia')
plt.xticks(rotation=45, ha='right')  # Ajusta la rotación de las etiquetas en el eje x si es necesario
plt.grid(axis='y', linestyle='--', alpha=0.7)  # Añade una cuadrícula horizontal
plt.show()

# Crear el diagrama de barras
plt.figure(figsize=(10, 6))  # Ajusta el tamaño del gráfico según tus preferencias
servicio.plot(kind='bar', color='brown')
# Personaliza el gráfico
plt.title('Diagrama de Barras de la Columna "Limpieza"')
plt.xlabel('Categorías')
plt.ylabel('Frecuencia')
plt.xticks(rotation=45, ha='right')  # Ajusta la rotación de las etiquetas en el eje x si es necesario
plt.grid(axis='y', linestyle='--', alpha=0.7)  # Añade una cuadrícula horizontal
plt.show()

# Crear el diagrama de barras
plt.figure(figsize=(10, 6))  # Ajusta el tamaño del gráfico según tus preferencias
servicio.plot(kind='bar', color='pink')
# Personaliza el gráfico
plt.title('Diagrama de Barras de la Columna "Calidad del sueño"')
plt.xlabel('Categorías')
plt.ylabel('Frecuencia')
plt.xticks(rotation=45, ha='right')  # Ajusta la rotación de las etiquetas en el eje x si es necesario
plt.grid(axis='y', linestyle='--', alpha=0.7)  # Añade una cuadrícula horizontal
plt.show()





#ANALISIS CLUSTER
# Elimina las filas que contienen el valor no válido
df1 = df1[df1['Total'] != '-']
df1 = df1[df1['Relación calidad-precio'] != '-']
X = df1[['Total', 'Relación calidad-precio']]


# Ajusta el modelo de K-Means con, por ejemplo, 3 clusters
modelo_kmeans = KMeans(n_clusters=3,  n_init=10)
modelo_kmeans.fit(X)


# Ahora puedes acceder y utilizar el modelo_kmeans
etiquetas_clusters = modelo_kmeans.labels_
centroides = modelo_kmeans.cluster_centers_


# Realiza acciones adicionales con el modelo si es necesario


# Visualiza los clusters
plt.scatter(X['Total'], X['Relación calidad-precio'], c=etiquetas_clusters, cmap='viridis')
plt.scatter(centroides[:, 0], centroides[:, 1], marker='X', s=200, linewidths=3, color='r')
plt.title('Análisis de Clusters')
plt.xlabel('Total')
plt.ylabel('Relación calidad-precio')
plt.show()
