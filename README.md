# Proyecto de Optimización: Cálculo de Rutas y de Carga de Camiones.

## 1. Flujo General del Trabajo.

1. **Lectura de datos desde la base de datos.**:
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

## 2. Descripción Detallada de los Scripts.

## Script 1: Optimización de la carga del camión.

### Importaciones y conexión a la base de datos.

- Se importan las librerías necesarias:
  - `pymysql` para conectar a la base de datos MySQL.
  - `pandas` para manipular la información en forma de DataFrame.
  - `pygad` para la parte de optimización con algoritmos genéticos.
  - `KMeans` de `sklearn.cluster` para el clustering de pedidos.
  - `folium` para la visualización en el mapa.
  - `numpy` para manejo de arreglos y funciones matemáticas.
- Se establece la conexión a la base de datos (`conn = pymysql.connect(...)).`

### Consulta de datos.

- Se obtienen los datos de los **camiones** (`truck`) y los **pedidos** (`order`) desde la base de datos, guardándolos en DataFrames de pandas:
  - `df_truck`: columnas `id`, `license_plate`, `max_mass`, `max_volume`.
  - `df_order`: columnas `id`, `maximum_permissible_mass`, `maximum_permissible_volume`, `longitude`, `latitude`.

### Funciones auxiliares.

- **`get_route_id(truck_id)`**: Dado el `id` de un camión, busca en la tabla `route` el `id` de la ruta asociada.
- **`update_order_route(order_id, route_id)`**: Actualiza el campo `route_id` de un pedido en la tabla `order`.
- **`can_fit_in_truck(cluster_df, truck_data, truck_id)`**: Verifica si **todo** un cluster de pedidos puede caber en el camión dado su uso actual (masa y volumen).
- **`fitness_function(...)`**: Función de aptitud (fitness) que se emplea en `pygad` para comprobar si la selección de clusters (con un vector binario) supera la capacidad del camión o no.
- **`verify_assignment(...)`**: Valida nuevamente si la suma de masa y volumen de un cluster sobrepasa la capacidad del camión.
- **`verify_total_truck_usage(...)`**: Verifica a nivel general si el uso acumulado (masa/volumen) de un camión no supera el 100%.
- **`update_truck_usage(...)`**: Acumula la masa y volumen utilizados por un camión.
- **`assign_cluster_to_truck(cluster_df, truck_id, truck_data, cluster_id)`**: Asigna de forma definitiva un cluster de pedidos al camión, si cumple con las restricciones.

### Preprocesamiento de datos y clustering.

- Se convierten a numéricos las columnas `maximum_permissible_mass` y `maximum_permissible_volume`.
- Se eliminan registros con valores nulos en dichas columnas.
- Se crea una columna `uploaded` para marcar si el pedido ha sido asignado (`True`/`False`).
- Se definen **8 centroides** manuales (predefinidos) en `cluster_centers`.
- Se aplica `KMeans` forzando la inicialización de los centroides (con `init=cluster_centers`).
- Cada pedido se clasifica en uno de los 8 clusters (`df_order['cluster']`).

### Estructura de datos para la asignación.

- Se agrupan los pedidos en un diccionario `clusters`, con la forma:
  ```python
  {
    'cluster_0': DataFrame con pedidos del cluster 0,
    'cluster_1': DataFrame con pedidos del cluster 1,
    ...
  }


## Script 2: Optimización de la Ruta descrita por el camión de reparto.

1. **Importación de librerías y configuración de variables.**  
   - Se importan las librerías necesarias: `requests`, `pandas`, `mysql.connector`, y funciones matemáticas para el cálculo de distancias (`radians`, `sin`, `cos`, etc.).  
   - Se definen constantes como la API Key de GraphHopper, la URL de la API, la ubicación del almacén y los parámetros de conexión a la base de datos.

2. **Función `calculate_distance(point1, point2)`.**  
   - Calcula la distancia entre dos puntos (latitud y longitud) usando la fórmula de la distancia haversine.  
   - Devuelve la distancia en metros.

3. **Función `greedy_route(start_point, locations)`.**  
   - Implementa un método codicioso para generar un recorrido inicial.  
   - Comienza desde el punto de inicio (almacén) y, en cada paso, selecciona el siguiente destino más cercano.  
   - Devuelve la lista de pedidos en el orden visitado.

4. **Consulta de rutas en la base de datos.**  
   - Se conecta a la base de datos y se buscan todos los `route_id` disponibles en la tabla `route`.  
   - Si no hay rutas, el script finaliza.

5. **Recorrido de cada ruta**  
   - Para cada `route_id`, se obtienen los pedidos asociados desde la tabla `order`.  
   - Se convierten los campos `Decimal` (lat, lng) a `float`.  
   - Se genera una ruta inicial con `greedy_route`.  
   - Se formatea la información (camión y servicios/pedidos) para enviarla a la API de GraphHopper.

6. **Llamada a la API de GraphHopper.**  
   - Se realiza un `POST` con `requests` enviando la información de vehículos y servicios.  
   - Si la respuesta es satisfactoria (código 200), se extrae el orden recomendado (`route_order`).

7. **Creación de un DataFrame con los resultados.**  
   - Con los resultados devueltos por la API, se construye un `DataFrame` que incluye el orden en que deben ser visitados los pedidos.  
   - Se ordena el `DataFrame` según la columna `id` (la secuencia de visita).

8. **Actualización de la base de datos.**  
   - Para cada pedido, se actualiza la columna `sequence` en la tabla `order` de la base de datos con el nuevo orden optimizado.

9. **Impresión de resultados y manejo de errores.**  
   - Se muestran en consola los `DataFrame` generados con el orden de visita.  
   - Si la API de GraphHopper falla, se muestra el error correspondiente.
