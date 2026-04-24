# -*- coding: utf-8 -*-

#import numpy as np
import random
import csv
models_dir='C:/Users/raque/OneDrive/Escritorio/prueba/general_opt_system/models/'

#print("Holaa")
# Class representing an electric vehicle
class ElectricVehicle:
    def __init__(self, id, distance, available_time_start, available_time_end, current_charge, battery_capacity, charge_speed, discharge_rate, assigned_std):
        self.id = id  # Vehicle identifier
        self.distance = distance  # Distance the vehicle needs to travel
        self.available_time_start = available_time_start  # Start time available for charging
        self.available_time_end = available_time_end  # End time available for charging
        self.current_charge = current_charge  # Current charge level
        self.battery_capacity = battery_capacity  # Battery capacity
        self.charge_speed = charge_speed  # Charging speed
        self.discharge_rate = discharge_rate  # KW discharged per 100 km (efficiency)
        self.assigned_std = assigned_std

    # Calculates and returns the time needed to charge the vehicle enough for the specified distance
    def get_charging_time(self):
        required_charge = self.distance * (self.discharge_rate/100)  # Calculation of the required charge for the distance to be covered

        # If the car does not have enough charge
        if self.current_charge < required_charge:
            required_charge -= self.current_charge  # Subtract the current charge
            return required_charge / self.charge_speed  # Time in hours required to charge the vehicle

        else:
            return 0

    # Calculates the time required to charge the vehicle
    def get_charging_cost(self, begin_time, end_time, hourly_prices):
        total_cost = 0
        current_time = begin_time
        while current_time < end_time:
            total_cost += self.charge_speed * hourly_prices[int(current_time)]
            current_time += 1
        return total_cost

    def __str__(self):
        return "Vehicle" + str(self.id)

    
# Class representing a element of solution list
class item_planlist:
    def __init__(self, vehiculo, begin_time, end_time, cargador ):
        self.vehiculo = vehiculo
        self.tiempo_inicio = begin_time
        self.tiempo_fin = end_time
        self.cargador = cargador
        
        
# Class representing a charger
class Cargador:
    def __init__(self, id):
        self.id = id
        self.periodos_ocupado = []

    def set_estado(self, hora_inicio, hora_fin):
        self.periodos_ocupado.append((hora_inicio, hora_fin))

    def eliminar_estado(self, hora_inicio, hora_fin):
        self.periodos_ocupado = [(inicio, fin) for inicio, fin in self.periodos_ocupado if (inicio, fin) != (hora_inicio, hora_fin)]

    def liberar_cargador(self):
        self.periodos_ocupado = []

    def esta_ocupado(self, hora_inicio, hora_fin):
        for inicio, fin in self.periodos_ocupado:
            if (hora_inicio>=inicio and hora_inicio<=fin) or (hora_fin>=inicio and hora_fin<=fin) :
                return True
        return False

    def tiene_horas_ocupadas(self):
        return bool(self.periodos_ocupado)

    
#Calculates timespan
def calcular_timespan(solucion):
    tiempo_inicio_mas_temprano = min(item.tiempo_inicio for item in solucion)
    tiempo_fin_mas_tardio = max(item.tiempo_fin for item in solucion)
    return tiempo_fin_mas_tardio - tiempo_inicio_mas_temprano



#######################ALGORITMO GENETICO############################


#Generates the initial population
def generar_poblacion_inicial(vehiculos, cargadores, population_size):

    #Set of solutions
    poblacion = []
    for _ in range(population_size):

        #Generate each solution
        solucion = []

        #Release chargers before generating a new solution.
        for i in range(len(cargadores)):
          cargadores[i].liberar_cargador()


        for vehiculo in vehiculos:
          #Check if the vehicle needs to be loaded
          if vehiculo.get_charging_time() > 0:

            tiempo_inicio = random.uniform(vehiculo.available_time_start, vehiculo.available_time_end-0.5)
            tiempo_fin = random.uniform(tiempo_inicio+0.5,tiempo_inicio+vehiculo.get_charging_time())
            tiempo_fin = min(tiempo_fin, vehiculo.available_time_end)  #Ensure that the loading time does not exceed the available time.

            cargador = None

            #Assign charger
            for i in range(len(cargadores)):
              if(cargadores[i].esta_ocupado(tiempo_inicio, tiempo_fin) == False):
                cargador = cargadores[i]
                cargadores[i].set_estado(tiempo_inicio, tiempo_fin)
                break

            if(cargador != None):
              it = item_planlist(vehiculo,tiempo_inicio,tiempo_fin,cargador)
              solucion.append(it)



        #dd solution to popilation
        poblacion.append(solucion)
    return poblacion


#Evaluate the population in terms of cost and timespan
def evaluar_poblacion(poblacion, hourly_prices, hora_llegada_media, hora_llegada_varianza, num_muestras=10):
    evaluaciones = []

    #For each solution in population
    for solucion in poblacion:
        coste_total = 0
        tiempo_total = 0
        timespan_ini = 0
        timespan_fin = 0
        acum_timespan_fin = 0
        acum_timespan_ini = 0

        #For each item (vehicle) in solution
        for i, item in enumerate(solucion):
            #Generate N random samples for the arrival time of each vehicle before the mean arrival time, hour_arrival_mean[i]
            #Usando numpy, descomentar para modificar
            #muestras_llegada = np.random.normal(item.vehiculo.available_time_start, hora_llegada_varianza[i], num_muestras)
            #Sin usar numpy
            muestras_llegada = [random.gauss(item.vehiculo.available_time_start, hora_llegada_varianza[i]) for _ in range(num_muestras)]
            coste_muestras = 0
            tiempo_muestras = 0


            #Generate mean arrival time following a probability distribution
            #muestras_salida = np.random.normal(item.vehiculo.available_time_end, hora_llegada_varianza[i], num_muestras)
            muestras_salida = [random.gauss(item.vehiculo.available_time_end, hora_llegada_varianza[i]) for _ in range(num_muestras)]
            
            #Calculate sum of elements
            suma = sum(muestras_salida)

            #Calculate number of elements
            cantidad_elementos = len(muestras_salida)

            #Calculate mean
            media_salida = suma / cantidad_elementos

            #Verify that no sample falls outside the vehicle's availability
            for muestra_llegada in muestras_llegada:
                tiempo_inicio = muestra_llegada
                tiempo_fin = min(muestra_llegada + item.vehiculo.get_charging_time(), media_salida)
                tiempo_carga = tiempo_fin - tiempo_inicio

                if tiempo_carga > 0:
                    coste_muestras += item.vehiculo.get_charging_cost(tiempo_inicio, tiempo_fin, hourly_prices)
                    tiempo_muestras += tiempo_carga

                if i == (len(solucion)-1):
                  acum_timespan_fin += tiempo_fin

                if i == 0:
                  acum_timespan_ini = tiempo_inicio

            coste_total += (coste_muestras / num_muestras)
            tiempo_total += tiempo_muestras / num_muestras
            timespan_fin = acum_timespan_fin / num_muestras
            timespan_ini = acum_timespan_ini / num_muestras

        timespan = timespan_fin - timespan_ini
        evaluaciones.append((solucion, coste_total, tiempo_total, timespan))

    return evaluaciones


#Select the best solutions for subsequent crossover
def seleccionar_padres(evaluaciones, num_padres):
    evaluaciones.sort(key=lambda x: (x[1], x[2]))
    padres = []
    for i in range(num_padres):
        padres.append(evaluaciones[i][0])
    
    print("Heyyy")
    return padres



#Crossover of parents
def cruzar_padres(padre1, padre2):
    hijo = []

    #Release the chargers before generating a new solution.
    for i in range(len(cargadores)):
        cargadores[i].liberar_cargador()

    for i in range(len(padre1)):

        tiempo_inicio_hijo = min(padre1[i].tiempo_inicio, padre2[i].tiempo_inicio)
        tiempo_fin_hijo = max(padre1[i].tiempo_fin, padre2[i].tiempo_fin)

        cargador = None
        #Assign charger
        for j in range(len(cargadores)):
            if(cargadores[j].esta_ocupado(tiempo_inicio_hijo, tiempo_fin_hijo) == False):
              cargador = cargadores[j]
              cargadores[j].set_estado(tiempo_inicio_hijo, tiempo_fin_hijo)
              break

        item = item_planlist(padre1[i].vehiculo, tiempo_inicio_hijo, tiempo_fin_hijo, cargador)

        if (cargador == None):
              item = item_planlist(padre1[i].vehiculo, padre1[i].tiempo_inicio, padre1[i].tiempo_fin, padre1[i].cargador)

        hijo.append(item)

    return hijo



#Introduce mutations
def mutar_hijo(hijo, mutation_rate):

    #Release chargers before generating a new solution.
    for i in range(len(cargadores)):
      cargadores[i].liberar_cargador()

    for i in range(len(hijo)):
        if random.uniform(0, 1) < mutation_rate:
            nuevo_tiempo_inicio = random.uniform(hijo[i].vehiculo.available_time_start, hijo[i].vehiculo.available_time_end - 0.5)
            nuevo_tiempo_fin = nuevo_tiempo_inicio + hijo[i].vehiculo.get_charging_time()
            nuevo_tiempo_fin = min(nuevo_tiempo_fin, hijo[i].vehiculo.available_time_end)

            if nuevo_tiempo_fin == nuevo_tiempo_inicio:
                nuevo_tiempo_fin = nuevo_tiempo_inicio + 0.5

            #Assign charger
            for j in range(len(cargadores)):
              if(cargadores[j].esta_ocupado(nuevo_tiempo_inicio, nuevo_tiempo_fin) == False):
                cargadores[j].set_estado(nuevo_tiempo_inicio, nuevo_tiempo_fin)
                hijo[i].tiempo_inicio = nuevo_tiempo_inicio
                hijo[i].tiempo_fin = nuevo_tiempo_fin
                hijo[i].cargador = cargadores[j]
                break

    return hijo


#Generación del modelo PRISM
def generate_prism_model(vehicles, filename="modelo"):
    
    consumer="vehicle"
    resource="charger"
    
    
    # Extraer cargadores de los vehículos
    chargers = {v['charger'] for v in vehicles}
    num_chargers = len(chargers)
    num_vehicles = len(vehicles)
    charge_rate = 10  # Puedes ajustar esto según sea necesario
    total_hours = 240  # Puedes ajustar esto según sea necesario

    # Inicio del modelo
    model = "dtmc\n\n"

    # Definir constantes del sistema
    model += "// Constantes del sistema\n"
    model += "const int N = {0};\n".format(num_chargers)
    model += "const int M = {0};\n".format(num_vehicles)
    model += "const int charge_rate = {0};\n\n".format(charge_rate)
    model += "const int MAX_BAT = 100;\n"

    # Vehicle states
    model += "const int NOT_AVAILABLE = 0;\n"
    model += "const int CHARGING = 1;\n"
    model += "const int CHARGED = 2;\n\n"

    # Charger states
    model += "const int FREE = 0;\n"
    model += "const int OCCUPIED = 1; \n\n"

    # Definir variables globales de tiempo y fase
    model += "// Seguimiento del tiempo\n"
    model += "const int TOTAL_HOURS = {0};\n".format(total_hours)
    model += "global current_time : [0..TOTAL_HOURS] init 0;\n"
    model += "global phase : [0..M] init 0;\n"
    model += "global start_time : [0..TOTAL_HOURS] init TOTAL_HOURS;\n"
    model += "global end_time : [0..TOTAL_HOURS] init 0;\n\n"

    # Definir el módulo de tiempo
    model += "module time\n"
    model += "    passturn_time: bool init false;\n\n"
    model += "    [] (phase=0) & (!passturn_time) & (current_time < TOTAL_HOURS) ->\n"
    model += "        (current_time' = current_time + 1) & (passturn_time' = true);\n\n"
    model += "    [] (passturn_time) -> (phase' = 1) & (passturn_time' = false);\n"
    model += "endmodule\n\n"

    # Módulos de vehículos
    for vehicle in vehicles:
        vehicle_id = vehicle['id']
        start_time = vehicle['start_time']
        end_time = vehicle['end_time']
        model += "module {1}{0}\n".format(vehicle_id, consumer)
        model += "    passturn_{1}{0}: bool init false;\n".format(vehicle_id)
        model += "    start_time_{1}{0} : [0..TOTAL_HOURS] init TOTAL_HOURS;\n".format(vehicle_id)
        model += "    end_time_{1}{0} : [0..TOTAL_HOURS] init 0;\n\n".format(vehicle_id)
        
        model += "    v{0}_update_timespan : bool init false;\n".format(vehicle_id)

        model += "    v{0}_charge_status : [NOT_AVAILABLE..CHARGED] init NOT_AVAILABLE;\n".format(vehicle_id)
        model += "    v{0}_battery : [0..100] init 20;  // Carga inicial\n".format(vehicle_id)
        model += "    v{0}_start_time : [0..TOTAL_HOURS] init {1};\n".format(vehicle_id, start_time)
        model += "    v{0}_end_time : [0..TOTAL_HOURS] init {1};\n\n".format(vehicle_id, end_time)

        model += "    [] (phase={0}) & (!passturn_{1}{0}) & (current_time < v{0}_start_time | v{0}_charge_status = CHARGED)->\n".format(vehicle_id)
        model += "        (passturn_{1}{0}' = true);\n\n".format(vehicle_id)

        model += "    [start_charge{0}] (phase={0}) & (!passturn_{1}{0}) & (current_time = v{0}_start_time & v{0}_charge_status != CHARGING & v{0}_battery < MAX_BAT)->\n".format(vehicle_id)
        model += "        (v{0}_charge_status' = CHARGING) & (v{0}_battery' = min(v{0}_battery + charge_rate, MAX_BAT)) & (passturn_{1}{0}' = true) & ".format(vehicle_id)
        model += "(start_time_{1}{0}' = min(start_time, current_time));\n\n".format(vehicle_id)
        
        model += "    [] (phase={0}) & (!passturn_{1}{0}) & (v{0}_charge_status = CHARGING) & (v{0}_battery < MAX_BAT) ".format(vehicle_id)
        model += "& (current_time > v{0}_start_time) & (current_time < v{0}_end_time) ->\n".format(vehicle_id)
        model += "        (v{0}_battery' = min(v{0}_battery + charge_rate, 100)) & (passturn_{1}{0}' = true);\n\n".format(vehicle_id)
        
        model += "    [release_charge{0}] (phase={0}) & (!passturn_{1}{0}) & (v{0}_battery = MAX_BAT) & (v{0}_charge_status = CHARGING) ->\n".format(vehicle_id)
        model += "        (v{0}_charge_status' = CHARGED) & (passturn_{1}{0}' = true) & (v{0}_update_timespan' = true);\n\n".format(vehicle_id)

        model += "    [release_charge{0}] (phase={0}) & (!passturn_{1}{0}) & (current_time >= v{0}_end_time) & (v{0}_battery < MAX_BAT) & (v{0}_charge_status = CHARGING)->\n".format(vehicle_id)
        model += "        (v{0}_charge_status' = NOT_AVAILABLE) & (passturn_{1}{0}' = true) & (end_time_{1}{0}' = max(end_time, current_time));\n\n".format(vehicle_id)
        
        model += "    [] (phase={0}) & (!passturn_{1}{0}) & (v{0}_battery = MAX_BAT) & (v{0}_charge_status != CHARGING) ->\n".format(vehicle_id)
        model += "        (passturn_{1}{0}' = true);\n\n".format(vehicle_id)

        model += "    [] (phase={0}) & (!passturn_{1}{0}) & (current_time >= v{0}_end_time) & (v{0}_battery < MAX_BAT) & (v{0}_charge_status != CHARGING)->\n ".format(vehicle_id)
        model += "        (passturn_{1}{0}' = true);\n\n ".format(vehicle_id)

        model += "    [] (passturn_{1}{0}) -> (phase' = (phase = M ? 0 : phase + 1)) & (passturn_{1}{0}' = false) & (end_time' = v{0}_update_timespan ? max(end_time, end_time_vehicle{0}): end_time) & (start_time' = min(start_time, start_time_vehicle{0}));\n".format(vehicle_id)
        model += "endmodule\n\n"

    # Módulos de cargadores
    for charger_id in chargers:
        model += "module charger{0}\n".format(charger_id)
        model += "    charger{0}_status : [FREE..OCCUPIED] init FREE;\n\n".format(charger_id)

        # Aquí se utilizan los vehículos que corresponden a cada cargador
        for vehicle in vehicles:
            if vehicle['charger'] == charger_id:
                vehicle_id = vehicle['id']
                model += "    [start_charge{0}] (charger{1}_status = FREE) -> (charger{1}_status' = OCCUPIED);\n".format(vehicle_id, charger_id)
                model += "    [release_charge{0}] (charger{1}_status = OCCUPIED) -> (charger{1}_status' = FREE);\n\n".format(vehicle_id, charger_id)

        model += "endmodule\n\n"

    # Recompensas
    model += "rewards \"total_time\"\n"
    for vehicle in vehicles:
        model += "    [] phase = 0 & v{0}_charge_status = CHARGING : 1;\n".format(vehicle['id'])
    model += "endrewards\n\n"

    model += "rewards \"total_cost\"\n"
    for vehicle in vehicles:
        model += "    [] phase = 0 & v{0}_charge_status = CHARGING & (current_time >= 0 & current_time < 60) : 2;\n".format(vehicle['id'])
        model += "    [] phase = 0 & v{0}_charge_status = CHARGING & (current_time >= 60 & current_time < 100) : 5;\n".format(vehicle['id'])
        model += "    [] phase = 0 & v{0}_charge_status = CHARGING & (current_time >= 100 & current_time < 180) : 8;\n".format(vehicle['id'])
        model += "    [] phase = 0 & v{0}_charge_status = CHARGING & (current_time >= 180 & current_time < 240) : 6;\n".format(vehicle['id'])
    model += "endrewards\n\n"

    model += "rewards \"charging_timespan\"\n"
    model += "    [] (end_time >= start_time) : end_time - start_time;\n"
    model += "endrewards\n"

    try:
        with open(models_dir+'modelop.prism', 'w') as file:
            file.write(model)
            #print(model)
        print("Modelo PRISM guardado en el archivo 'modelo.txt'.")
    except Exception as e:
        print("Error al guardar el archivo: {}".format(e))


    print("Modelo PRISM guardado en el archivo '{0}.txt'.".format(filename))



import time

# Parameters of the genetic algorithm
population_size = 10
num_generations = 50
num_padres = 3
mutation_rate = 0.2

# Generate chargers list
cargadores = [Cargador(1), Cargador(2), Cargador(3), Cargador(4), Cargador(5), Cargador(6) ]

# Example of vehicles and prices per hour - Arrival and departure time with mean and variance
# Example of vehicles
'''vehiculo1 = ElectricVehicle(1, 100, 9, 10, 2, 80, 10, 5)
vehiculo2 = ElectricVehicle(2, 200, 8, 9, 3, 100, 8, 4)
vehiculo3 = ElectricVehicle(3, 150, 12, 13, 1, 120, 12, 6)
vehiculo4 = ElectricVehicle(4, 200, 12, 13, 1, 120, 12, 6)
vehiculo5 = ElectricVehicle(5, 100, 14, 17, 2, 80, 10, 5)
vehiculo6 = ElectricVehicle(6, 200, 13, 18, 3, 100, 8, 4)
vehiculo7 = ElectricVehicle(7, 150, 12, 20, 1, 120, 12, 6)
vehiculo8 = ElectricVehicle(8, 200, 15, 22, 1, 120, 12, 6)
vehiculo9 = ElectricVehicle(9, 100, 9, 10, 2, 80, 10, 5)
vehiculo10 = ElectricVehicle(10, 200, 8, 20, 3, 100, 8, 4)
vehiculo11 = ElectricVehicle(11, 150, 12, 18, 1, 120, 12, 6)
vehiculo12 = ElectricVehicle(12, 200, 14, 20, 1, 120, 12, 6)
vehiculo13 = ElectricVehicle(13, 100, 21, 22, 2, 80, 10, 5)
vehiculo14 = ElectricVehicle(14, 200, 22, 23, 3, 100, 8, 4)
vehiculo15 = ElectricVehicle(15, 150, 20, 23, 1, 120, 12, 6)
vehiculo16 = ElectricVehicle(16, 200, 15, 22, 1, 120, 12, 6)
vehiculo17 = ElectricVehicle(17, 100, 9, 20, 2, 80, 10, 5)
vehiculo18 = ElectricVehicle(18, 200, 6, 10, 3, 100, 8, 4)
vehiculo19 = ElectricVehicle(19, 150, 4, 9, 1, 120, 12, 6)
vehiculo20 = ElectricVehicle(20, 200, 17, 23, 1, 120, 12, 6)

vehiculos = [vehiculo1, vehiculo2, vehiculo3, vehiculo4, vehiculo5, vehiculo6, vehiculo7, vehiculo8, vehiculo9, vehiculo10]'''


#Create vehicle objects from file
def create_vehicles_from_file(file_name):
    vehicles = []
    with open(file_name, 'r') as file:
        lines = file.readlines()
        for line in lines:
            data = line.strip().split(', ')
            if len(data) == 8:
                id, distance, available_time_start, available_time_end, current_charge, battery_capacity, charge_speed, discharge_rate = map(float, data)
                vehicle = ElectricVehicle(id, distance, available_time_start, available_time_end, current_charge, battery_capacity, charge_speed, discharge_rate)
                vehicles.append(vehicle)
    return vehicles

#file_name = 'C:\\Users\\raque\\OneDrive\\Escritorio\\prueba\\general_opt_system\\data\\vehiculos.txt'
#vehiculos = create_vehicles_from_file(file_name)

def create_vehicles_from_csv_py2(file_name):
    vehicles = []
    with open(file_name, 'rb') as csvfile:  # 'rb' para Python 2
        reader = csv.reader(csvfile)
        next(reader, None)  # saltar cabecera
        for row in reader:
            # Asegurarse de que tiene al menos 9 columnas (las 8 originales + std)
            if len(row) >= 9:
                id, distance, available_time_start, available_time_end, current_charge, battery_capacity, charge_speed, discharge_rate, assigned_std = row[:9]
                vehicle = ElectricVehicle(
                    float(id), float(distance), float(available_time_start), float(available_time_end),
                    float(current_charge), float(battery_capacity), float(charge_speed), float(discharge_rate),
                    float(assigned_std)
                )
                vehicles.append(vehicle)
    return vehicles

# Ejemplo de uso
file_name = 'vehicles_clustered_with_std.csv'
vehiculos = create_vehicles_from_csv_py2(file_name)

# Example of vehicles and prices per hour - Arrival and departure time with mean and variance
hora_llegada_media = [9, 8, 12, 12, 14, 13, 12, 15, 9, 8]
hora_llegada_varianza = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

#, 12, 14, 21, 22, 20, 15, 9, 6, 4, 17    , 1, 1, 1, 1, 1, 1, 1, 1, 1, 1   2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2 , 2, 2, 2    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1

hourly_prices = [0.1, 0.2, 0.15, 0.25, 0.2, 0.15, 0.1, 0.1, 0.2, 0.3, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.1, 0.2, 0.3, 0.35, 0.3, 0.25, 0.2, 0.15]

start_time = time.time()
# Generate initial population
poblacion = generar_poblacion_inicial(vehiculos, cargadores, population_size)

#Print population - Uncomment if necessary
'''print("POBLACION")
for sol in poblacion:
  for it in sol:
    print("identificador vehiculo: "+ str(it.vehiculo.id))
    print("inicio "+str(it.tiempo_inicio))
    print("fin "+str(it.tiempo_fin))
    print("cargador "+str(it.cargador.id))
    print("")

  print("-------------------------------------------------")'''

'''evaluaciones=evaluar_poblacion(poblacion,hourly_prices)
padres = seleccionar_padres(evaluaciones, num_padres)
padre1 = random.choice(padres)
padre2 = random.choice(padres)
hijo = cruzar_padres(padre1, padre2)
hijo = mutar_hijo(hijo, mutation_rate)'''



# Run genetic algorithm
for generation in range(num_generations):
    evaluaciones = evaluar_poblacion(poblacion, hourly_prices, hora_llegada_media, hora_llegada_varianza)
    padres = seleccionar_padres(evaluaciones, num_padres)

    nueva_poblacion = []
    for i in range(population_size):
        padre1 = random.choice(padres)
        padre2 = random.choice(padres)
        hijo = cruzar_padres(padre1, padre2)
        hijo_mutado = mutar_hijo(hijo, mutation_rate)
        nueva_poblacion.append(hijo_mutado)

    poblacion = nueva_poblacion

# Get the best solution found
evaluaciones = evaluar_poblacion(poblacion, hourly_prices, hora_llegada_media, hora_llegada_varianza)
evaluaciones.sort(key=lambda x: (x[1], x[2]))
mejor_sol = evaluaciones[0]
end_time = time.time()

print("#########################################################################")

# Print the solution
print("Coste total: " + str(mejor_sol[1]))
print("Tiempo total: " + str(mejor_sol[2]))
print("Timespan " + str(mejor_sol[3]) )
print("Planificación:")

""" for item in mejor_sol[0]:
  print(item.vehiculo.id)
  print(item.tiempo_inicio)
  print(item.tiempo_fin)
  print(item.cargador.id)
  print("") """
  
  
list_model_vehicles = []

for item in mejor_sol[0]:
    vehicle_data = {
        'id': int(item.vehiculo.id),
        'start_time': int(item.tiempo_inicio*10),
        'end_time': int(item.tiempo_fin*10),
        'charger': int(item.cargador.id)
    }
    list_model_vehicles.append(vehicle_data)

# Print vehicles list
for vehicle in list_model_vehicles:
    print(vehicle)
    
# Generate PRISM model
generate_prism_model(list_model_vehicles)

execution_time = end_time - start_time
#print(f"Tiempo de ejecución: {execution_time * 1000:.2f} milisegundos")
print("Tiempo total: " + str(execution_time))



# STATISTICS TO EVALUATE SOLUTION QUALITY - Uncomment to execute
# file_name = 'C:\\Users\\raque\\OneDrive\\Escritorio\\prueba\\general_opt_system\\data\\vehiculosprueba.txt'
# vehiculos2 = create_vehicles_from_file(file_name)

# i=0
# cont=0
# vehiculos_noc=0
# tiempo_real_carga=0
# coste_real_total = 0
# coste_real=0
# coste_penalizacion=0
# timespan_penalizacion=0
# for item in mejor_sol[0]:

#   if item.vehiculo.id == vehiculos2[i].id:

#     if item.tiempo_inicio<vehiculos2[i].available_time_start:
#       cont=cont+1
#       tiempo_real_carga = item.tiempo_fin - vehiculos2[i].available_time_start
#       tiempo_real_ini= vehiculos2[i].available_time_start

#       if vehiculos[i].get_charging_time()>tiempo_real_carga:
#           vehiculos_noc=vehiculos_noc+1
#           timespan_penalizacion=vehiculos[i].get_charging_time()-tiempo_real_carga
#           #Para los vehiculos que por llegar tarde no se carguen lo suficiente, se añade una penalización sumando el coste de cargar lo que les falta en el momento más caro del día
#           #Tiempo que queda x velocidad de carga x precio de hora más cara
#           coste_penalizacion=coste_penalizacion+(timespan_penalizacion*vehiculos[i].charge_speed)*0.35

#     else:
#       tiempo_real_carga=item.tiempo_fin-item.tiempo_inicio
#       tiempo_real_ini=item.tiempo_inicio

#     coste_real=vehiculos[i].get_charging_cost(item.tiempo_inicio, item.tiempo_fin, hourly_prices)


#     timespan_inicio=0
#     timespan_fin=0

#     if i==0:
#       timespan_inicio=tiempo_real_ini

#     else: timespan_fin=item.tiempo_fin+timespan_penalizacion

#   i=i+1
#   coste_real_total = coste_real_total + coste_real/2 ++coste_penalizacion
#   timespan=timespan_fin-timespan_inicio

# print("Número de intentos de carga y que no esté el vehículo")
# print(cont)

# print("Número de vehículos que no se cargan lo suficiente")
# print(vehiculos_noc)

# print("Coste real total")
# print(coste_real_total)

# print("Timespan real")
# print(timespan)