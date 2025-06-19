# clustering_analysis.py

import pandas as pd
import seaborn as sb
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Cargar los datos
datos = pd.read_csv("housing.csv")
print(datos.head())

# Visualización geográfica del valor de vivienda
sb.scatterplot(
    x="latitude", y="longitude",
    data=datos,
    hue="median_house_value",
    palette="coolwarm"
)
plt.title("Distribución de valores de viviendas")
plt.show()

# Visualización de ingresos medianos
sb.scatterplot(
    x="latitude", y="longitude",
    data=datos,
    hue="median_income",
    size="median_income",
    sizes=(20, 200),
    legend=False
)
plt.title("Distribución por ingreso mediano")
plt.show()

# Seleccionar columnas
x = datos[["latitude", "longitude", "median_income"]]

# Aplicar clustering KMeans
modelo = KMeans(n_clusters=6, random_state=42)
x["segmento_economico"] = modelo.fit_predict(x)

# Conteo por segmento
print(x["segmento_economico"].value_counts())

# Visualización por segmento económico
sb.scatterplot(
    x="latitude", y="longitude",
    data=x,
    hue="segmento_economico",
    palette="bright"
)
plt.title("Segmentación económica por ubicación")
plt.show()

# Conteo de segmentos
sb.countplot(x="segmento_economico", data=x)
plt.title("Cantidad por segmento económico")
plt.show()

# Ingresos promedio por segmento
print(x.groupby("segmento_economico")["median_income"].mean())
