<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8" />
    <title>README - ItinerarIA</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0 20px;
            line-height: 1.6;
        }
        h1, h2, h3, h4 {
            margin-top: 1em;
        }
        pre {
            background-color: #f4f4f4;
            padding: 10px;
            overflow-x: auto;
            border-radius: 5px;
            border: 1px solid #ccc;
        }
        code {
            font-family: Consolas, "Liberation Mono", Courier, monospace;
            color: #333;
        }
        .section {
            margin-bottom: 2em;
        }
        .section ul {
            list-style-type: disc;
            padding-left: 40px;
        }
        .important {
            background-color: #fffae6;
            border-left: 4px solid #ffdb4d;
            padding: 10px;
            margin: 10px 0;
        }
    </style>
</head>
<body>

    <h1>ItinerarIA - Optimización y Asignación de Pedidos</h1>
    <p>
        <strong>ItinerarIA</strong> es un proyecto que implementa dos scripts en Python para:
    </p>
    <ul>
        <li><strong>Optimizar</strong> rutas de entrega utilizando la API de GraphHopper.</li>
        <li><strong>Asignar</strong> pedidos a camiones, teniendo en cuenta restricciones de volumen y masa, mediante un enfoque de clustering y un algoritmo genético.</li>
    </ul>
    
    <div class="section">
        <h2>Índice</h2>
        <ol>
            <li><a href="#descripcion-general">Descripción General</a></li>
            <li><a href="#requisitos">Requisitos e Instalación</a></li>
            <li><a href="#uso-del-proyecto">Uso del Proyecto</a></li>
            <li><a href="#script1">Documentación: Script 1 (Optimización de Ruta)</a></li>
            <li><a href="#script2">Documentación: Script 2 (Asignación de Pedidos a Camiones)</a></li>
            <li><a href="#contacto">Contacto</a></li>
        </ol>
    </div>

    <div class="section" id="descripcion-general">
        <h2>1. Descripción General</h2>
        <p>
            El proyecto <strong>ItinerarIA</strong> consta de dos scripts principales que realizan tareas complementarias:
        </p>
        <ul>
            <li>
                El <strong>Script 1</strong> se conecta a una base de datos MySQL para obtener rutas y pedidos, 
                y hace uso de la API de <em>GraphHopper</em> para la optimización de la secuencia de entrega. 
                Finalmente, actualiza la secuencia de entrega en la base de datos.
            </li>
            <li>
                El <strong>Script 2</strong> realiza la asignación de los pedidos a camiones, 
                considerando la capacidad máxima en masa y en volumen de cada camión. 
                Aplica clustering (K-Means con centroides predefinidos) y un algoritmo genético 
                (usando <em>pygad</em>) para optimizar la asignación.
            </li>
        </ul>
    </div>

    <div class="section" id="requisitos">
        <h2>2. Requisitos e Instalación</h2>
        <ul>
            <li><strong>Python 3.x</strong></li>
            <li>Librerías de Python:
                <ul>
                    <li>requests</li>
                    <li>pandas</li>
                    <li>mysql-connector-python o mysqlclient (según tu preferencia)</li>
                    <li>pymysql</li>
                    <li>pygad</li>
                    <li>scikit-learn (para KMeans)</li>
                    <li>folium (para visualización en mapa)</li>
                </ul>
            </li>
            <li>Servidor <strong>MySQL</strong> (base de datos con las tablas <code>route</code>, <code>order</code>, <code>truck</code>, etc.)</li>
            <li>Cuenta y clave de API <strong>GraphHopper</strong>.</li>
        </ul>
        <p>
            Para instalar las dependencias de Python:
        </p>
        <pre><code>pip install requests pandas mysql-connector-python pymysql pygad scikit-learn folium</code></pre>
    </div>

    <div class="section" id="uso-del-proyecto">
        <h2>3. Uso del Proyecto</h2>
        <ol>
            <li>Asegúrate de que tu base de datos <strong>MySQL</strong> está configurada y contiene las tablas requeridas.</li>
            <li>Actualiza la información de conexión a la base de datos en los scripts (<em>DB_CONFIG</em> en el <em>Script 1</em> y el método de conexión en el <em>Script 2</em>).</li>
            <li>Proporciona tu <strong>API_KEY</strong> de GraphHopper en el <em>Script 1</em>.</li>
            <li>Ejecuta primero el <em>Script 2</em> (asignación de pedidos a camiones), para actualizar la columna <code>route_id</code> de la tabla <code>order</code>.</li>
            <li>Posteriormente, ejecuta el <em>Script 1</em> para optimizar la secuencia de entrega en cada ruta y actualizar la columna <code>sequence</code> en la tabla <code>order</code>.</li>
        </ol>
    </div>

    <div class="section" id="script1">
        <h2>4. Documentación: Script 1 (Optimización de Ruta)</h2>

        <p>
            Este script se encarga de:
        </p>
        <ul>
            <li>Conectarse a la base de datos para obtener todas las rutas (tabla <code>route</code>).</li>
            <li>Para cada ruta, obtiene los pedidos asociados en la tabla <code>order</code>.</li>
            <li>Los pedidos se ordenan primero con un recorrido "greedy" para obtener un orden inicial basándose en la distancia mínima al siguiente punto.</li>
            <li>Posteriormente, se envía una petición a la API de <strong>GraphHopper</strong> para optimizar el orden de visita de dichos pedidos.</li>
            <li>Se registran los resultados de la optimización en la columna <code>sequence</code> de la tabla <code>order</code>.</li>
        </ul>

        <h3>Fragmentos Relevantes</h3>

        <p><strong>1) Configuración de la API y Funciones</strong></p>
        <pre><code class="language-python">
# Configuración
API_KEY = "TU_API_KEY_DE_GRAPHOPPER"
API_URL_OPTIMIZATION = "https://graphhopper.com/api/1/vrp"
API_URL_ROUTE = "https://graphhopper.com/api/1/route"

warehouse_location = {"lat": 27.96683841473653, "lng": -15.392203774815524}

# Función para calcular distancia (geográfica) entre dos puntos
def calculate_distance(point1, point2):
    ...
</code></pre>
        <p>
            Se define la ubicación del almacén (<em>warehouse_location</em>) y las URLs para la API de GraphHopper. 
            Además, se implementa la función <code>calculate_distance()</code> para el cálculo de la distancia entre dos coordenadas usando la fórmula de Haversine.
        </p>

        <p><strong>2) Orden "Greedy" de los Puntos</strong></p>
        <pre><code class="language-python">
def greedy_route(start_point, locations):
    ...
</code></pre>
        <p>
            El método <em>greedy_route</em> toma un <em>start_point</em> (en este caso, el almacén) y una lista de ubicaciones 
            para determinar un orden de visita inicial seleccionando siempre el siguiente punto más cercano.
        </p>

        <p><strong>3) Consulta de Rutas y Pedidos en la BD</strong></p>
        <pre><code class="language-python">
with mysql.connector.connect(**DB_CONFIG) as conn:
    with conn.cursor(dictionary=True) as cursor:
        cursor.execute("SELECT id FROM `route`")
        routes = cursor.fetchall()
    ...
</code></pre>
        <p>
            Se consulta la tabla <code>route</code> para obtener todas las rutas disponibles. 
            Luego, para cada ruta, se obtienen los pedidos (tabla <code>order</code>) con el <code>route_id</code> correspondiente.
        </p>

        <p><strong>4) Llamada a la API de GraphHopper y Actualización de la BD</strong></p>
        <pre><code class="language-python">
payload = {"vehicles": [vehicle], "services": services}
response = requests.post(f"{API_URL_OPTIMIZATION}?key={API_KEY}", json=payload, headers=headers)

if response.status_code == 200:
    data = response.json()
    ...
</code></pre>
        <p>
            Se construye un <em>payload</em> con la lista de vehículos (en este caso, un vehículo por ruta) y la lista de servicios (pedidos). 
            GraphHopper retorna la solución óptima (o más cercana a la óptima) en la respuesta JSON. 
            Finalmente, se actualiza la columna <code>sequence</code> en la tabla <code>order</code> con el orden calculado.
        </p>

        <h3>Código Completo del Script 1</h3>
        <pre><code class="language-python">
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

# Función para calcular la distancia entre dos puntos (Haversine)
def calculate_distance(point1, point2):
    R = 6371e3
    lat1, lon1 = radians(point1[0]), radians(point1[1])
    lat2, lon2 = radians(point2[0]), radians(point2[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

# Ordenar los puntos en base al recorrido "greedy"
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

#-------------------------------------------------------------------------
# Procesamiento principal
with mysql.connector.connect(**DB_CONFIG) as conn:
    # Obtener IDs de rutas
    with conn.cursor(dictionary=True) as cursor:
        cursor.execute("SELECT id FROM `route`")
        routes = cursor.fetchall()

    if not routes:
        print("No hay rutas disponibles en la base de datos.")
        exit()

    # Procesar cada ruta
    for route in routes:
        ROUTE_ID = route['id']
        print(f"Procesando pedidos para la ruta con route_id = {ROUTE_ID}...")

        # Obtener los pedidos de la BD
        with conn.cursor(dictionary=True) as cursor:
            cursor.execute("""
                SELECT id, latitude AS lat, longitude AS lng
                FROM `order`
                WHERE route_id = %s
            """, (ROUTE_ID,))
            order_locations = cursor.fetchall()

        if not order_locations:
            print(f"No hay pedidos asignados a la ruta con route_id = {ROUTE_ID}.")
            continue

        # Convertir Decimal a float
        for loc in order_locations:
            loc["lat"] = float(loc["lat"])
            loc["lng"] = float(loc["lng"])

        # Crear la ruta optimizada localmente (greedy)
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

        response = requests.post(f"{API_URL_OPTIMIZATION}?key={API_KEY}", json=payload, headers=headers)

        if response.status_code == 200:
            data = response.json()

            # Obtener el orden de la ruta optimizada
            route_order = []
            for route_data in data["solution"]["routes"]:
                for activity in route_data["activities"]:
                    if activity["type"] == "service":
                        route_order.append(activity["id"])

            # Construir DataFrame para actualizar la BD
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

            print(df)

            # Actualizar columna "sequence" en la tabla "order"
            for _, row in df.iterrows():
                with conn.cursor() as cursor:
                    cursor.execute("""
                        UPDATE `order`
                        SET sequence = %s
                        WHERE id = %s
                    """, (row["id"], row["id_order"]))
                    conn.commit()

            print(f"Columna 'sequence' actualizada para los pedidos de la ruta {ROUTE_ID}.")
        else:
            print(f"Error en la API de GraphHopper para route_id {ROUTE_ID}: {response.status_code} - {response.text}")

    # Ejemplo adicional de final de archivo (puede ser redundante)
    # ...
</code></pre>
    </div>

    <div class="section" id="script2">
        <h2>5. Documentación: Script 2 (Asignación de Pedidos a Camiones)</h2>
        <p>
            Este script maneja la lógica de <strong>asignar pedidos a los camiones</strong> considerando las capacidades de masa y volumen. 
            La información se obtiene de la base de datos MySQL y después de realizar la asignación, se actualiza el <code>route_id</code> 
            de cada pedido para indicar a qué camión/ruta pertenece.
        </p>

        <h3>Proceso General</h3>
        <ol>
            <li>Conexión a MySQL y obtención de los registros de camiones (<code>truck</code>) y pedidos (<code>order</code>).</li>
            <li>Definición de una función (<code>get_route_id</code>) para relacionar cada camión (<code>truck_id</code>) con su ruta (<code>route_id</code> en la tabla <code>route</code>).</li>
            <li>Aplicación de <strong>K-Means</strong> con centroides específicos para agrupar geográficamente los pedidos en <code>n_clusters</code>.</li>
            <li>Uso de un <strong>algoritmo genético</strong> (con la librería <code>pygad</code>) para decidir qué clusters se asignan a cada camión según su capacidad.</li>
            <li>Actualización de la columna <code>route_id</code> de los pedidos asignados en la tabla <code>order</code>.</li>
            <li>Generación de un <strong>mapa (HTML)</strong> con <code>folium</code> para visualizar los clusters y los pedidos asignados.</li>
        </ol>

        <h3>Código Completo del Script 2</h3>
        <pre><code class="language-python">
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

# Preparación de datos en df_order
df_order["maximum_permissible_mass"] = pd.to_numeric(df_order["maximum_permissible_mass"], errors="coerce")
df_order["maximum_permissible_volume"] = pd.to_numeric(df_order["maximum_permissible_volume"], errors="coerce")
df_order.dropna(subset=["maximum_permissible_mass", "maximum_permissible_volume"], inplace=True)
df_order['uploaded'] = False

# Definición del número de clusters y centroides iniciales
n_clusters = 8
cluster_centers = np.array([
    (28.135035564504964, -15.43209759947092),
    (28.11873238338806, -15.52326563195904),
    (28.14414695029059, -15.655172960469848),
    (28.100333635665432, -15.705940715919775),
    (28.039781912565754, -15.572606885537912),
    (27.99972252231642, -15.41705962589178),
    (27.91787907070111, -15.432363893330333),
    (27.770627079086285, -15.605982396663174)
])

# Aplicar K-Means con centroides específicos
kmeans = KMeans(n_clusters=n_clusters, init=cluster_centers, n_init=1, random_state=666)
df_order['cluster'] = kmeans.fit_predict(df_order[["latitude", "longitude"]])

# Crear estructura de datos para los clusters
clusters = {f'cluster_{i}': df_order[df_order['cluster'] == i].copy() for i in range(n_clusters)}

# Variables para seguimiento de asignaciones
orders_in_trucks = {}
volume_truck_used = {}
truck_usage = {truck['license_plate']: {'mass': 0, 'volume': 0} for _, truck in df_truck.iterrows()}

# Funciones de verificación de capacidad
def can_fit_in_truck(cluster_df, truck_data, truck_id):
    total_mass = cluster_df['maximum_permissible_mass'].sum()
    total_volume = cluster_df['maximum_permissible_volume'].sum()

    current_usage = truck_usage[truck_id]
    remaining_mass_capacity = truck_data['max_mass'] - current_usage['mass']
    remaining_volume_capacity = truck_data['max_volume'] - current_usage['volume']

    return (total_mass <= remaining_mass_capacity * 0.99 and
            total_volume <= remaining_volume_capacity * 0.99)

def fitness_function(ga_instance, solution, solution_idx, remaining_clusters, truck_data, truck_id):
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
    usage = truck_usage[truck_id]
    truck_data = df_truck[df_truck['license_plate'] == truck_id].iloc[0]

    mass_percentage = (usage['mass'] / truck_data['max_mass']) * 100
    volume_percentage = (usage['volume'] / truck_data['max_volume']) * 100

    return mass_percentage <= 100 and volume_percentage <= 100

def update_truck_usage(truck_id, mass, volume):
    truck_usage[truck_id]['mass'] += mass
    truck_usage[truck_id]['volume'] += volume

def assign_cluster_to_truck(cluster_df, truck_id, truck_data, cluster_id):
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

# Proceso principal
available_clusters = list(range(n_clusters))
trucks_list = df_truck['license_plate'].tolist()
current_truck_index = 0

while available_clusters and current_truck_index < len(trucks_list):
    current_truck_id = trucks_list[current_truck_index]
    current_truck = df_truck[df_truck['license_plate'] == current_truck_id].iloc[0]

    first_cluster = available_clusters[0]
    first_cluster_df = clusters[f'cluster_{first_cluster}']

    # Verificar si cabe el primer cluster entero
    if can_fit_in_truck(first_cluster_df, current_truck, current_truck_id):
        assign_cluster_to_truck(first_cluster_df, current_truck_id, current_truck, first_cluster)
        available_clusters.remove(first_cluster)

        # Asignación de clusters restantes con GA
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
                        print(f"Índice {idx} fuera de rango. Longitud actual: {len(available_clusters)}")
                        continue

                    cluster_id = available_clusters[idx]
                    cluster_df = clusters[f'cluster_{cluster_id}']
                    if assign_cluster_to_truck(cluster_df, current_truck_id, current_truck, cluster_id):
                        available_clusters.remove(cluster_id)

    current_truck_index += 1

# Crear mapa de clusters con folium
map_center = [df_order['latitude'].mean(), df_order['longitude'].mean()]
map_clusters = folium.Map(location=map_center, zoom_start=12)

colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'lightblue', 'pink']

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
            popup=(f"ID: {row['id']}<br>Volume: {row['maximum_permissible_volume']:.2f}"
                   f"<br>Mass: {row['maximum_permissible_mass']:.2f}<br>Status: {status}")
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

# Mostrar resultados finales y actualizar la DB con route_id
print("\nResumen de asignaciones por camión y cluster:")
for truck_key, orders in orders_in_trucks.items():
    truck_license = truck_key.split('_')[1]
    truck_id = df_truck[df_truck['license_plate'] == truck_license]['id'].iloc[0]
    route_id = get_route_id(truck_id)

    if route_id:
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
</code></pre>

        <p>
            El mapa generado (<code>clusters_map.html</code>) se guardará en el mismo directorio del script 
            y mostrará la ubicación de cada pedido (color según <strong>cluster</strong>) y los centroides de cada cluster.
        </p>
    </div>

    <div class="section" id="contacto">
        <h2>6. Contacto</h2>
        <p>
            Si tienes dudas o sugerencias sobre este proyecto, puedes contactar a:
            <br />
            <strong>Ana</strong> – <em>Desarrolladora</em>
        </p>
    </div>

    <hr />
    <p><em>© 2023 - ItinerarIA</em></p>
</body>
</html>
