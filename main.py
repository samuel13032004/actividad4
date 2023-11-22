import pandas as pd
import statsmodels.api as sm
from factor_analyzer import FactorAnalyzer #analisis factorial
from sklearn.cluster import KMeans
from sklearn import metrics
import matplotlib.pyplot as plt #graficos
from matplotlib.widgets import Button  # Importa la clase Button desde el módulo widgets

from sklearn.decomposition import FactorAnalysis
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# Especifica la ruta de tu archivo Excel
archivo_excel = 'FMCC Actividad 4 nuevo.xlsx'
archivo_excel_analisis_cluster = 'FMCC Actividad 4 nuevo.xlsx'

# Carga el archivo Excel en un DataFrame de pandas
df = pd.read_excel(archivo_excel)
df1 = pd.read_excel(archivo_excel_analisis_cluster)

df1.replace('-', pd.NA, inplace=True)# para reemplazar guiones por NAs
df1.dropna(inplace=True)# para eliminar filas con NA

# Eliminar la columna 'Ciudad' en el mismo DataFrame
df1.drop('Opinión Cualitativa', axis=1, inplace=True)

# # Mostrar el DataFrame después de eliminar la columna
# print("\nDataFrame después de eliminar la columna 'Opinión Cualitativa':")
# print(df1)
#
# # Accede a las columnas del DataFrame
# columnas = df1.columns
#
# # Imprime las columnas
# print("Columnas en el archivo Excel:")
# for columna in columnas:
#     print(columna)
# print("\n")

##################################################################################################
# # ANALISIS DE FRECUENCIA
# df_frecuencia = df1
# total = df_frecuencia['Total'].value_counts()
# relacion_calidad_precio = df_frecuencia['Relación calidad-precio'].value_counts()
# ubicacion = df_frecuencia['Ubicación'].value_counts()
# servicio = df_frecuencia['Servicio'].value_counts()
# habitaciones = df_frecuencia['Habitaciones'].value_counts()
# limpieza = df_frecuencia['Limpieza'].value_counts()
# calidad_sueño = df_frecuencia['Calidad del sueño'].value_counts()
# # Imprime las frecuencias
# print("Análisis de Frecuencias:")
# print(total)
# print(relacion_calidad_precio)
# print(ubicacion)
# print(servicio)
# print(habitaciones)
# print(limpieza)
# print(calidad_sueño)
# print("\n")
#
# # Lista de columnas sobre las que deseas hacer gráficos
# columnas_interes = ['Total', 'Relación calidad-precio', 'Ubicación', 'Servicio', 'Limpieza', 'Calidad del sueño']
#
# # Colores diferentes para cada gráfico
# colores = ['grey', 'green', 'yellow', 'blue', 'brown', 'pink']
#
# # Crear un diccionario para almacenar las figuras
# figuras = {}
#
#
# # Función para mostrar el gráfico y conectar al evento de teclado
# def mostrar_grafico(columna, color):
#     figuras[columna] = plt.figure(figsize=(10, 6))
#     df_frecuencia[columna].value_counts().plot(kind='bar', color=color)
#     plt.title(f'Diagrama de Barras de la Columna "{columna}"')
#     plt.xlabel('Categorías')
#     plt.ylabel('Frecuencia')
#     plt.xticks(rotation=45, ha='right')
#     plt.grid(axis='y', linestyle='--', alpha=0.7)
#     plt.gcf().canvas.mpl_connect('key_press_event', on_key)
#
#
# # Función para cerrar todas las figuras al presionar la tecla 'Esc'
# def on_key(event):
#     if event.key == 'escape':
#         for _, figura in figuras.items():
#             plt.close(figura)
#
#
# # Iterar sobre las columnas e invocar la función para mostrar el gráfico con colores diferentes
# for columna, color in zip(columnas_interes, colores):
#     mostrar_grafico(columna, color)
#
# # Mostrar todas las figuras
# plt.show()

##############################################################################################################
#
#
# # REGRESION MÚLTIPLE
# # Reemplazar el valor "-" por NaN
# df.replace('-', pd.NA, inplace=True)
#
# # Convertir columnas a tipo numérico
# df[['Relación calidad-precio', 'Ubicación', 'Servicio', 'Habitaciones', 'Limpieza', 'Calidad del sueño']] = (
#     df[['Relación calidad-precio', 'Ubicación', 'Servicio', 'Habitaciones', 'Limpieza', 'Calidad del sueño']]
#     .apply(pd.to_numeric, errors='coerce'))
#
# # Eliminar filas que contienen NaN en alguna de las columnas de interés
# df.dropna(subset=['Relación calidad-precio', 'Ubicación', 'Servicio', 'Habitaciones', 'Limpieza', 'Calidad del sueño'],
#           inplace=True)
#
# # Dividir los datos en características (X) y variable objetivo (y)
# X = df[['Ubicación', 'Servicio', 'Habitaciones', 'Limpieza', 'Calidad del sueño']]
# y = df['Relación calidad-precio']
#
# #comprobar que las variables son numericas
# #print(df.dtypes)
# #print(df.isnull().sum())
#
# # Dividir los datos en conjuntos de entrenamiento y prueba
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
#
# ## Para un analisis mas estadistico
# # Agrega una columna de intercepto a X (La constante beta_0)
# X = sm.add_constant(X)
#
# # Crea el modelo de regresión lineal con statsmodels
# modelo_stats = sm.OLS(y, X).fit()
#
# # Imprime un resumen del modelo
# print(modelo_stats.summary())
#
# # Realizar predicciones en el conjunto de prueba
# predicciones_stats = modelo_stats.predict(sm.add_constant(X_test))
#
# # Evaluar el rendimiento del modelo
# print("\nSe procede a evaluar el rendimiento del modelo a partir de predicciones")
# print('Error absoluto medio:', metrics.mean_absolute_error(y_test, predicciones_stats))
# print('Error cuadrático medio:', metrics.mean_squared_error(y_test, predicciones_stats))
# print('Raíz del error cuadrático medio:', metrics.mean_squared_error(y_test, predicciones_stats, squared=False))
#
# # Función para cerrar la ventana de visualización al presionar la tecla 'Escape'
# def on_key(event):
#     if event.key == 'escape':
#         plt.close()
#
# # Conectar la función al evento de teclado
# fig, ax = plt.subplots()
# fig.canvas.mpl_connect('key_press_event', on_key)
#
# # Visualizar las predicciones
# ax.scatter(y_test, predicciones_stats)
# ax.set_xlabel('Relación calidad-precio real')
# ax.set_ylabel('Relación calidad-precio predicha')
# plt.show()
#
###############################################################################################
# ANALISIS FACTORIAL
# Crear una copia del DataFrame original
# Función para cerrar todas las figuras al presionar 'esc'
def on_key(event):
    if event.key == 'escape':
        plt.close('all')

df_factorial = df.copy()

# Seleccionar las columnas relevantes para el análisis factorial
columnas_analisis = ['Ubicación', 'Servicio', 'Habitaciones', 'Limpieza', 'Calidad del sueño']

# Crear un DataFrame con las columnas seleccionadas y hacer una copia explícita
df_factor = df_factorial[['Relación calidad-precio'] + columnas_analisis].copy()

# Reemplazar valores faltantes
df_factor.replace('-', pd.NA, inplace=True)
df_factor.dropna(inplace=True)

# Realizar el análisis factorial para cada par de "Relación calidad-precio" con otras variables
for columna in columnas_analisis:
    # Crear un DataFrame para el par actual
    df_par = pd.DataFrame({
        'Relación calidad-precio': df_factor['Relación calidad-precio'],
        columna: df_factor[columna]
    })

    # Realizar el análisis factorial
    analizador_factor = FactorAnalyzer(n_factors=1, rotation='varimax')
    analizador_factor.fit(df_par)

    # Obtener las cargas factoriales
    cargas_factoriales = analizador_factor.loadings_

    # Imprimir las cargas factoriales para el par actual
    print(f"Cargas Factoriales para '{columna}':")
    print(cargas_factoriales)

    # Graficar las cargas factoriales para el par actual
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(range(2), cargas_factoriales[0])
    ax.set_xticks(range(2))
    ax.set_xticklabels(['Relación calidad-precio', columna])
    ax.set_title(f'Cargas Factoriales para "{columna}"')

    # Conectar la función de cierre con el evento 'esc'
    fig.canvas.mpl_connect('key_press_event', on_key)

# Mostrar todas las figuras
plt.show()

#################################################################################################
# #ANALISIS CLUSTER
# # Elimina las filas que contienen el valor no válido
# df1 = df1[df1['Total'] != '-']
# df1 = df1[df1['Relación calidad-precio'] != '-']
# X = df1[['Total', 'Relación calidad-precio']]
#
#
# # Ajusta el modelo de K-Means con, por ejemplo, 3 clusters
# modelo_kmeans = KMeans(n_clusters=3,  n_init=10)
# modelo_kmeans.fit(X)
#
#
# # Ahora puedes acceder y utilizar el modelo_kmeans
# etiquetas_clusters = modelo_kmeans.labels_
# centroides = modelo_kmeans.cluster_centers_
#
#
# # Realiza acciones adicionales con el modelo si es necesario
#
#
# # Visualiza los clusters
# plt.scatter(X['Total'], X['Relación calidad-precio'], c=etiquetas_clusters, cmap='viridis')
# plt.scatter(centroides[:, 0], centroides[:, 1], marker='X', s=200, linewidths=3, color='r')
# plt.title('Análisis de Clusters')
# plt.xlabel('Total')
# plt.ylabel('Relación calidad-precio')
# #plt.show()
#
#
