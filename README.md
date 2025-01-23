# Proyecto de Optimización de Rutas y Asignación de Pedidos

## 1. Flujo General del Trabajo

1. **Lectura de datos desde la base de datos**:
   - Se extrae información relevante (rutas, pedidos, camiones, etc.) de la base de datos.
   
2. **Script 1**:
   - **Objetivo**: Asignar los pedidos existentes a camiones específicos de forma que se optimice el uso del espacio y masa disponible de cada camión.
   - **Procesos principales**:
     1. Clasificar los pedidos en clusters (grupos) utilizando K-Means con centroides predefinidos.  
     2. Verificar la capacidad de cada camión (masa y volumen).  
     3. Asignar cada cluster de pedidos a un camión apropiado y actualizar la base de datos con esta asignación.  
     4. Generar un mapa HTML mostrando la distribución geográfica de los pedidos, sus estados (asignado / no asignado) y los centroides de cada cluster.
   
3. **Script 2**: 
   - **Objetivo**: Optimizar el orden de entrega de pedidos asociados a cada ruta utilizando la API de GraphHopper y un método de recorrido "greedy" (codicioso) para generar el orden inicial de visitas.  
   - **Procesos principales**:
     1. Calcular la distancia entre puntos (almacén y pedidos).  
     2. Generar un recorrido básico (orden codicioso) para cada ruta a partir de la ubicación del almacén.  
     3. Enviar los datos a la API de GraphHopper para obtener un orden más refinado.  
     4. Guardar la secuencia final de pedidos optimizada en la base de datos.
---

## 2. Descripción Detallada de los Scripts

### Script 1: Optimización de Secuencia de Pedidos y Actualización de la Base de Datos

#### Explicación Paso a Paso

1. **Importación de librerías y configuración de variables**  
   - Se importan las librerías necesarias: `requests`, `pandas`, `mysql.connector`, y funciones matemáticas para el cálculo de distancias (`radians`, `sin`, `cos`, etc.).  
   - Se definen constantes como la API Key de GraphHopper, la URL de la API, la ubicación del almacén y los parámetros de conexión a la base de datos.

2. **Función `calculate_distance(point1, point2)`**  
   - Calcula la distancia entre dos puntos (latitud y longitud) usando la fórmula de la distancia haversine.  
   - Devuelve la distancia en metros.

3. **Función `greedy_route(start_point, locations)`**  
   - Implementa un método codicioso para generar un recorrido inicial.  
   - Comienza desde el punto de inicio (almacén) y, en cada paso, selecciona el siguiente destino más cercano.  
   - Devuelve la lista de pedidos en el orden visitado.

4. **Consulta de rutas en la base de datos**  
   - Se conecta a la base de datos y se buscan todos los `route_id` disponibles en la tabla `route`.  
   - Si no hay rutas, el script finaliza.

5. **Recorrido de cada ruta**  
   - Para cada `route_id`, se obtienen los pedidos asociados desde la tabla `order`.  
   - Se convierten los campos `Decimal` (lat, lng) a `float`.  
   - Se genera una ruta inicial con `greedy_route`.  
   - Se formatea la información (camión y servicios/pedidos) para enviarla a la API de GraphHopper.

6. **Llamada a la API de GraphHopper**  
   - Se realiza un `POST` con `requests` enviando la información de vehículos y servicios.  
   - Si la respuesta es satisfactoria (código 200), se extrae el orden recomendado (`route_order`).

7. **Creación de un DataFrame con los resultados**  
   - Con los resultados devueltos por la API, se construye un `DataFrame` que incluye el orden en que deben ser visitados los pedidos.  
   - Se ordena el `DataFrame` según la columna `id` (la secuencia de visita).

8. **Actualización de la base de datos**  
   - Para cada pedido, se actualiza la columna `sequence` en la tabla `order` de la base de datos con el nuevo orden optimizado.

9. **Impresión de resultados y manejo de errores**  
   - Se muestran en consola los `DataFrame` generados con el orden de visita.  
   - Si la API de GraphHopper falla, se muestra el error correspondiente.

#### Código Completo del Script 1

```python
import requests
import pandas as pd
import mysql.connector
from math import radians, sin, cos, sqrt, atan2

# Configuración inicial
API_KEY = "35ba8f2c-ecde-4176-bd17-203259ebebef"
API_URL_OPTIMIZATION = "https://graphhopper.com/api/1/vrp"
API_URL_ROUTE = "https://graphhopper.com/api/1/route"

warehouse_location = {"lat": 27.96683841473653, "lng": -15.392203774815524}

# Configuración de la base de datos
DB_CONFIG = {
    "host": "localhost",
    "user": "ana",
    "password": "ana",
    "database": "itinerarIA"
}

# Función para calcular la distancia entre dos puntos
def calculate_distance(point1, point2):
    R = 6371e3
    lat1, lon1 = radians(point1[0]), radians(point1[1])
    lat2, lon2 = radians(point2[0]), radians(point2[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

# Ordenar los puntos en base al recorrido codicioso
def greedy_route(start_point, locations):
    remaining = locations[:]
    route = []
    current_point = start_point
    while remaining:
        next_point = min(
            remaining,
            key=lambda loc: calculate_distance(
                (current_point["lat"], current_point["lng"]),
                (loc["lat"], loc["lng"])
            )
        )
        route.append(next_point)
        remaining.remove(next_point)
        current_point = next_point
    return route

#**************************************************************************
# Obtener todas las rutas disponibles de la tabla "route"
with mysql.connector.connect(**DB_CONFIG) as conn:
    with conn.cursor(dictionary=True) as cursor:
        cursor.execute("SELECT id FROM route")
        routes = cursor.fetchall()

    if not routes:
        print("No hay rutas disponibles en la base de datos.")
        exit()

    # Procesar cada ruta
    for route in routes:
        ROUTE_ID = route['id']
        print(f"Procesando pedidos para la ruta con route_id = {ROUTE_ID}...")

        # Obtener los pedidos asociados a la ruta actual
        with conn.cursor(dictionary=True) as cursor:
            cursor.execute("""
                SELECT id, latitude AS lat, longitude AS lng
                FROM order
                WHERE route_id = %s
            """, (ROUTE_ID,))
            order_locations = cursor.fetchall()

        if not order_locations:
            print(f"No hay pedidos asignados a la ruta con route_id = {ROUTE_ID}.")
            continue  # Pasar a la siguiente ruta

        # Convertir Decimal a float
        for loc in order_locations:
            loc["lat"] = float(loc["lat"])
            loc["lng"] = float(loc["lng"])

        # Crear la ruta optimizada con método codicioso
        sorted_locations = greedy_route(warehouse_location, order_locations)

        # Preparar datos para GraphHopper
        vehicle = {
            "vehicle_id": f"vehicle_{ROUTE_ID}",
            "start_address": {
                "location_id": "warehouse",
                "lon": warehouse_location["lng"],
                "lat": warehouse_location["lat"]
            }
        }

        services = []
        for loc in sorted_locations:
            services.append({
                "id": str(loc["id"]),
                "address": {
                    "location_id": str(loc["id"]),
                    "lon": loc["lng"],
                    "lat": loc["lat"]
                }
            })

        payload = {"vehicles": [vehicle], "services": services}
        headers = {"Content-Type": "application/json"}

        response = requests.post(
            f"{API_URL_OPTIMIZATION}?key={API_KEY}",
            json=payload,
            headers=headers
        )

        if response.status_code == 200:
            data = response.json()
            route_order = []

            for route in data["solution"]["routes"]:
                for activity in route["activities"]:
                    if activity["type"] == "service":
                        route_order.append(activity["id"])

            # Crear un DataFrame con los resultados
            df_data = []
            for loc in sorted_locations:
                df_data.append({
                    "id": route_order.index(str(loc["id"])) + 1 if str(loc["id"]) in route_order else None,
                    "id_order": loc["id"],
                    "latitude": loc["lat"],
                    "longitude": loc["lng"]
                })

            df = pd.DataFrame(df_data)
            # Ordenar el DataFrame por la columna 'id'
            df = df.sort_values(by="id").reset_index(drop=True)
            print(df)

            # Actualizar los resultados en la columna "sequence" de la tabla "order"
            for _, row in df.iterrows():
                with conn.cursor() as cursor:
                    cursor.execute("""
                        UPDATE order
                        SET sequence = %s
                        WHERE id = %s
                    """, (row["id"], row["id_order"]))
                conn.commit()

            print(f"Columna 'sequence' actualizada para los pedidos de la ruta {ROUTE_ID}.")

        else:
            print(f"Error en la API de GraphHopper para route_id {ROUTE_ID}: {response.status_code} - {response.text}")

        #*************************************************************************
        # Repetición de la lógica de optimización (al parecer por comprobaciones adicionales)
        for loc in order_locations:
            loc["lat"] = float(loc["lat"])
            loc["lng"] = float(loc["lng"])

        sorted_locations = greedy_route(warehouse_location, order_locations)

        vehicle = {
            "vehicle_id": "van_1",
            "start_address": {
                "location_id": "warehouse",
                "lon": warehouse_location["lng"],
                "lat": warehouse_location["lat"]
            }
        }

        services = []
        for loc in sorted_locations:
            services.append({
                "id": str(loc["id"]),
                "address": {
                    "location_id": str(loc["id"]),
                    "lon": loc["lng"],
                    "lat": loc["lat"]
                }
            })

        payload = {"vehicles": [vehicle], "services": services}
        headers = {"Content-Type": "application/json"}
        response = requests.post(
            f"{API_URL_OPTIMIZATION}?key={API_KEY}",
            json=payload,
            headers=headers
        )

        if response.status_code == 200:
            data = response.json()
            route_order = []
            for route in data["solution"]["routes"]:
                for activity in route["activities"]:
                    if activity["type"] == "service":
                        route_order.append(activity["id"])

            df_data = []
            for loc in sorted_locations:
                df_data.append({
                    "id": route_order.index(str(loc["id"])) + 1 if str(loc["id"]) in route_order else None,
                    "id_order": loc["id"],
                    "latitude": loc["lat"],
                    "longitude": loc["lng"]
                })

            df = pd.DataFrame(df_data)
            df = df.sort_values(by="id").reset_index(drop=True)

            for _, row in df.iterrows():
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE order
                    SET sequence = %s
                    WHERE id = %s
                """, (row["id"], row["id_order"]))
                conn.commit()
                cursor.close()

            conn.close()
            print(f"Columna 'sequence' actualizada para los pedidos de la ruta {ROUTE_ID}.")
        else:
            print(f"Error en la API de GraphHopper: {response.status_code} - {response.text}")

        print(payload['vehicles'])


```


markdown
# Script 2: Asignación de Pedidos a Camiones según Capacidad y Ubicación

## Explicación Paso a Paso

### Importaciones y conexión a la base de datos

- Se importan las librerías necesarias:
  - `pymysql` para conectar a la base de datos MySQL.
  - `pandas` para manipular la información en forma de DataFrame.
  - `pygad` para la parte de optimización con algoritmos genéticos.
  - `KMeans` de `sklearn.cluster` para el clustering de pedidos.
  - `folium` para la visualización en el mapa.
  - `numpy` para manejo de arreglos y funciones matemáticas.
- Se establece la conexión a la base de datos (`conn = pymysql.connect(...)).`

### Consulta de datos

- Se obtienen los datos de los **camiones** (`truck`) y los **pedidos** (`order`) desde la base de datos, guardándolos en DataFrames de pandas:
  - `df_truck`: columnas `id`, `license_plate`, `max_mass`, `max_volume`.
  - `df_order`: columnas `id`, `maximum_permissible_mass`, `maximum_permissible_volume`, `longitude`, `latitude`.

### Funciones auxiliares

- **`get_route_id(truck_id)`**: Dado el `id` de un camión, busca en la tabla `route` el `id` de la ruta asociada.
- **`update_order_route(order_id, route_id)`**: Actualiza el campo `route_id` de un pedido en la tabla `order`.
- **`can_fit_in_truck(cluster_df, truck_data, truck_id)`**: Verifica si **todo** un cluster de pedidos puede caber en el camión dado su uso actual (masa y volumen).
- **`fitness_function(...)`**: Función de aptitud (fitness) que se emplea en `pygad` para comprobar si la selección de clusters (con un vector binario) supera la capacidad del camión o no.
- **`verify_assignment(...)`**: Valida nuevamente si la suma de masa y volumen de un cluster sobrepasa la capacidad del camión.
- **`verify_total_truck_usage(...)`**: Verifica a nivel general si el uso acumulado (masa/volumen) de un camión no supera el 100%.
- **`update_truck_usage(...)`**: Acumula la masa y volumen utilizados por un camión.
- **`assign_cluster_to_truck(cluster_df, truck_id, truck_data, cluster_id)`**: Asigna de forma definitiva un cluster de pedidos al camión, si cumple con las restricciones.

### Preprocesamiento de datos y clustering

- Se convierten a numéricos las columnas `maximum_permissible_mass` y `maximum_permissible_volume`.
- Se eliminan registros con valores nulos en dichas columnas.
- Se crea una columna `uploaded` para marcar si el pedido ha sido asignado (`True`/`False`).
- Se definen **8 centroides** manuales (predefinidos) en `cluster_centers`.
- Se aplica `KMeans` forzando la inicialización de los centroides (con `init=cluster_centers`).
- Cada pedido se clasifica en uno de los 8 clusters (`df_order['cluster']`).

### Estructura de datos para la asignación

- Se agrupan los pedidos en un diccionario `clusters`, con la forma:
  ```python
  {
    'cluster_0': DataFrame con pedidos del cluster 0,
    'cluster_1': DataFrame con pedidos del cluster 1,
    ...
  }
