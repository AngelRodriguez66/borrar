# Proyecto de Optimización: Cálculo de Rutas y de Carga de Camiones.

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

## Script 1: Optimización de la carga del camión.

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

#### Código Completo del Script 1
```python

import pymysql
import pandas as pd
import pygad
from sklearn.cluster import KMeans
import folium
import numpy as np

# Conexión a la base de datos MySQL
conn = pymysql.connect(
    host="localhost",
    user="ana",
    password="ana",
    database="itinerarIA"
)

cursor = conn.cursor()

# Consultas a la base de datos
query_truck = "SELECT id, license_plate, max_mass, max_volume FROM truck"
cursor.execute(query_truck)
df_truck = pd.DataFrame(cursor.fetchall(), columns=["id", "license_plate", "max_mass", "max_volume"])

query_order = "SELECT id, maximum_permissible_mass, maximum_permissible_volume, longitude, latitude FROM `order`"
cursor.execute(query_order)
df_order = pd.DataFrame(cursor.fetchall(),
                        columns=["id", "maximum_permissible_mass", "maximum_permissible_volume", "longitude",
                                 "latitude"])

# Función para obtener el route_id basado en el truck_id
def get_route_id(truck_id):
    query = "SELECT id FROM route WHERE truck_id = %s"
    cursor.execute(query, (truck_id,))
    result = cursor.fetchone()
    return result[0] if result else None

# Función para actualizar el route_id en la tabla order
def update_order_route(order_id, route_id):
    query = "UPDATE `order` SET route_id = %s WHERE id = %s"
    cursor.execute(query, (route_id, order_id))
    conn.commit()

# Preparación de datos
df_order["maximum_permissible_mass"] = pd.to_numeric(df_order["maximum_permissible_mass"], errors="coerce")
df_order["maximum_permissible_volume"] = pd.to_numeric(df_order["maximum_permissible_volume"], errors="coerce")
df_order.dropna(subset=["maximum_permissible_mass", "maximum_permissible_volume"], inplace=True)
df_order['uploaded'] = False

# Parámetros del clustering con centroides específicos
n_clusters = 8
cluster_centers = np.array([
    (28.135035564504964, -15.43209759947092),  # Sitio A
    (28.11873238338806, -15.52326563195904),   # Sitio B
    (28.14414695029059, -15.655172960469848),  # Sitio C
    (28.100333635665432, -15.705940715919775), # Sitio D
    (28.039781912565754, -15.572606885537912), # Sitio E
    (27.99972252231642, -15.41705962589178),  # Sitio F
    (27.91787907070111, -15.432363893330333),  # Sitio G
    (27.770627079086285, -15.605982396663174)  # Sitio H
])

# Aplicar K-Means con centroides iniciales específicos
kmeans = KMeans(n_clusters=n_clusters, init=cluster_centers, n_init=1, random_state=666)
df_order['cluster'] = kmeans.fit_predict(df_order[["latitude", "longitude"]])

# Crear diccionarios para resultados
clusters = {f'cluster_{i}': df_order[df_order['cluster'] == i].copy() for i in range(n_clusters)}
orders_in_trucks = {}
volume_truck_used = {}

# Diccionario para seguimiento del uso de cada camión
truck_usage = {truck['license_plate']: {'mass': 0, 'volume': 0} for _, truck in df_truck.iterrows()}

def can_fit_in_truck(cluster_df, truck_data, truck_id):
    """
    Verifica si un cluster completo cabe en el camión considerando el uso actual
    """
    total_mass = cluster_df['maximum_permissible_mass'].sum()
    total_volume = cluster_df['maximum_permissible_volume'].sum()

    current_usage = truck_usage[truck_id]
    remaining_mass_capacity = truck_data['max_mass'] - current_usage['mass']
    remaining_volume_capacity = truck_data['max_volume'] - current_usage['volume']

    return (total_mass <= remaining_mass_capacity * 0.99 and
            total_volume <= remaining_volume_capacity * 0.99)

def fitness_function(ga_instance, solution, solution_idx, remaining_clusters, truck_data, truck_id):
    """
    Función de fitness con verificación más estricta
    """
    total_mass = truck_usage[truck_id]['mass']
    total_volume = truck_usage[truck_id]['volume']

    for i, use_cluster in enumerate(solution):
        if use_cluster == 1:
            cluster_df = remaining_clusters[i]
            cluster_mass = cluster_df['maximum_permissible_mass'].sum()
            cluster_volume = cluster_df['maximum_permissible_volume'].sum()

            if (total_mass + cluster_mass > truck_data['max_mass'] * 0.99 or
                    total_volume + cluster_volume > truck_data['max_volume'] * 0.99):
                return 0

            total_mass += cluster_mass
            total_volume += cluster_volume

    volume_utilization = total_volume / truck_data['max_volume']
    mass_utilization = total_mass / truck_data['max_mass']

    return (volume_utilization + mass_utilization) / 2

def verify_assignment(cluster_df, truck_id, truck_data):
    """
    Verificación final antes de asignar un cluster
    """
    total_mass = cluster_df['maximum_permissible_mass'].sum()
    total_volume = cluster_df['maximum_permissible_volume'].sum()

    current_usage = truck_usage[truck_id]
    final_mass = current_usage['mass'] + total_mass
    final_volume = current_usage['volume'] + total_volume

    if final_mass > truck_data['max_mass'] or final_volume > truck_data['max_volume']:
        print(f"¡Advertencia! Asignación rechazada para camión {truck_id}:")
        print(f"Masa final: {final_mass}/{truck_data['max_mass']}")
        print(f"Volumen final: {final_volume}/{truck_data['max_volume']}")
        return False
    return True

def verify_total_truck_usage(truck_id, df_truck):
    """
    Verifica que el uso total del camión no exceda sus límites
    """
    usage = truck_usage[truck_id]
    truck_data = df_truck[df_truck['license_plate'] == truck_id].iloc[0]

    mass_percentage = (usage['mass'] / truck_data['max_mass']) * 100
    volume_percentage = (usage['volume'] / truck_data['max_volume']) * 100

    return mass_percentage <= 100 and volume_percentage <= 100

def update_truck_usage(truck_id, mass, volume):
    """
    Actualiza el uso del camión
    """
    truck_usage[truck_id]['mass'] += mass
    truck_usage[truck_id]['volume'] += volume

def assign_cluster_to_truck(cluster_df, truck_id, truck_data, cluster_id):
    """
    Asigna un cluster completo a un camión
    """
    total_mass = cluster_df['maximum_permissible_mass'].sum()
    total_volume = cluster_df['maximum_permissible_volume'].sum()

    current_usage = truck_usage[truck_id]
    if (current_usage['mass'] + total_mass > truck_data['max_mass'] or
            current_usage['volume'] + total_volume > truck_data['max_volume']):
        print(f"¡Error! El cluster {cluster_id} excede la capacidad del camión {truck_id}.")
        return False

    update_truck_usage(truck_id, total_mass, total_volume)

    truck_key = f"Truck_{truck_id}_cluster_{cluster_id}"
    orders_in_trucks[truck_key] = cluster_df['id'].tolist()
    volume_truck_used[truck_key] = {
        'volume_used': total_volume,
        'volume_capacity': truck_data['max_volume'],
        'mass_used': total_mass,
        'mass_capacity': truck_data['max_mass']
    }

    df_order.loc[df_order['cluster'] == cluster_id, 'uploaded'] = True
    return True

# Proceso principal de asignación
available_clusters = list(range(n_clusters))
trucks_list = df_truck['license_plate'].tolist()
current_truck_index = 0

while available_clusters and current_truck_index < len(trucks_list):
    current_truck_id = trucks_list[current_truck_index]
    current_truck = df_truck[df_truck['license_plate'] == current_truck_id].iloc[0]

    first_cluster = available_clusters[0]
    first_cluster_df = clusters[f'cluster_{first_cluster}']

    if can_fit_in_truck(first_cluster_df, current_truck, current_truck_id):
        assign_cluster_to_truck(first_cluster_df, current_truck_id, current_truck, first_cluster)
        available_clusters.remove(first_cluster)

        remaining_clusters_data = [clusters[f'cluster_{i}'] for i in available_clusters]

        if remaining_clusters_data:
            ga_instance = pygad.GA(
                num_generations=100,
                num_parents_mating=5,
                fitness_func=lambda ga, sol, idx: fitness_function(
                    ga, sol, idx, remaining_clusters_data, current_truck, current_truck_id
                ),
                sol_per_pop=20,
                num_genes=len(remaining_clusters_data),
                gene_space=[0, 1],
                crossover_type="single_point",
                mutation_type="random",
                mutation_probability=0.1
            )

            ga_instance.run()
            solution, solution_fitness, _ = ga_instance.best_solution()

            if solution_fitness > 0:
                selected_indices = [i for i, val in enumerate(solution) if val == 1]
                selected_indices = [idx for idx in selected_indices if idx < len(available_clusters)]

                for idx in selected_indices:
                    if idx >= len(available_clusters):
                        print(f"Índice {idx} fuera de rango. Longitud actual de available_clusters: {len(available_clusters)}")
                        continue

                    cluster_id = available_clusters[idx]
                    cluster_df = clusters[f'cluster_{cluster_id}']
                    if assign_cluster_to_truck(cluster_df, current_truck_id, current_truck, cluster_id):
                        available_clusters.remove(cluster_id)

    current_truck_index += 1

# Crear visualización
map_center = [df_order['latitude'].mean(), df_order['longitude'].mean()]
map_clusters = folium.Map(location=map_center, zoom_start=12)

colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'lightblue', 'pink']

# Agregar marcadores de pedidos
for cluster_id in range(n_clusters):
    cluster_color = colors[cluster_id]
    cluster_data = df_order[df_order['cluster'] == cluster_id]

    for _, row in cluster_data.iterrows():
        status = "Asignado" if row['uploaded'] else "No asignado"
        folium.CircleMarker(
            location=(row['latitude'], row['longitude']),
            radius=5,
            color=cluster_color,
            fill=True,
            fill_opacity=0.7,
            popup=f"ID: {row['id']}<br>Volume: {row['maximum_permissible_volume']:.2f}<br>Mass: {row['maximum_permissible_mass']:.2f}<br>Status: {status}"
        ).add_to(map_clusters)

# Agregar centroides
for i, center in enumerate(cluster_centers):
    folium.CircleMarker(
        location=(center[0], center[1]),
        radius=8,
        color='black',
        fill=True,
        popup=f'Centroide {i}',
        weight=2
    ).add_to(map_clusters)

map_clusters.save("clusters_map.html")

# Imprimir resultados y actualizar route_id
print("\nResumen de asignaciones por camión y cluster:")
for truck_key, orders in orders_in_trucks.items():
    # Extraer la matrícula del camión del truck_key
    truck_license = truck_key.split('_')[1]
    
    # Obtener el truck_id
    truck_id = df_truck[df_truck['license_plate'] == truck_license]['id'].iloc[0]
    
    # Obtener el route_id correspondiente
    route_id = get_route_id(truck_id)
    
    if route_id:
        # Actualizar los pedidos con el route_id
        for order_id in orders:
            update_order_route(order_id, route_id)
        
        print(f"\n{truck_key}:")
        print(f"Truck ID: {truck_id}")
        print(f"Route ID: {route_id}")
        print(f"Pedidos asignados: {orders}")
        
        usage = volume_truck_used[truck_key]
        print(f"Volumen utilizado: {usage['volume_used']:.2f}/{usage['volume_capacity']:.2f} "
              f"({(usage['volume_used'] / usage['volume_capacity'] * 100):.1f}%)")
        print(f"Masa utilizada: {usage['mass_used']:.2f}/{usage['mass_capacity']:.2f} "
              f"({(usage['mass_used'] / usage['mass_capacity'] * 100):.1f}%)")
    else:
        print(f"\nAdvertencia: No se encontró ruta para el camión {truck_license}")

print("\nUso total por camión:")
for truck_id, usage in truck_usage.items():
    truck_data = df_truck[df_truck['license_plate'] == truck_id].iloc[0]
    max_volume = truck_data['max_volume']
    max_mass = truck_data['max_mass']

    volume_percentage = (usage['volume'] / max_volume) * 100
    mass_percentage = (usage['mass'] / max_mass) * 100

    print(f"\nCamión {truck_id}:")
    print(f"Volumen total utilizado: {volume_percentage:.1f}%")
    print(f"Masa total utilizada: {mass_percentage:.1f}%")

# Verificar pedidos no asignados
unassigned_orders = df_order[~df_order['uploaded']]['id'].tolist()
if unassigned_orders:
    print("\nPedidos no asignados:", unassigned_orders)
else:
    print("\nTodos los pedidos fueron asignados exitosamente")

# Cerrar conexión
conn.close()

```



## Script 2: Optimización de la Ruta descrita por el camión de reparto.

## Explicación Paso a Paso

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

#### Código Completo del Script 2

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
