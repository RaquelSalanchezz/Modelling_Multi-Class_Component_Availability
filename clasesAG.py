# -*- coding: utf-8 -*-
from __future__ import division, print_function

import generated_classes
import random

models_dir='C:/Users/raque/OneDrive/Escritorio/prueba/general_opt_system/models/'
config_dir="C:\\Users\\raque\\OneDrive\\Escritorio\\prueba\\general_opt_system\\config_files\\"

# Class representing a element of solution list
class item_planlist:
    def __init__(self, consumer, begin_time, end_time, resource ):
        self.consumer = consumer
        self.begin_time = begin_time
        self.end_time = end_time
        self.resource = resource
    
#Calculates timespan
def calcular_timespan(solucion):
    first_begin_time = min(item.begin_time for item in solucion)
    last_end_time = max(item.end_time for item in solucion)
    return last_end_time - first_begin_time



####################### GENETIC ALGORITHM ############################


#Generates the initial population
def generate_initial_population(consumers, resources, population_size):

    population = []
    for _ in range(population_size):

        solution = []

        for i in range(len(resources)):
          resources[i].release()

        for consumer in consumers:
          if consumer.get_process_time() > 0:

            initial_time = random.uniform(consumer.available_time_start, consumer.available_time_end-0.5)
            final_time = random.uniform(initial_time+0.5, initial_time+consumer.get_process_time())
            final_time = min(final_time, consumer.available_time_end)

            resource = None

            for i in range(len(resources)):
              if(resources[i].is_occupied(initial_time, final_time) == False):
                resource = resources[i]
                resources[i].set_state(initial_time, final_time)
                break

            if(resource != None):
              it = item_planlist(consumer, initial_time, final_time, resource)
              solution.append(it)

        population.append(solution)
        
    return population


# Evaluate the population in terms of cost and timespan.
# Each consumer uses its own assigned_std for uncertainty sampling.
# arrival_time_variance kept as optional fallback for backwards compatibility.
def evaluate_fitness(population, hourly_prices, arrival_time_media=None, arrival_time_variance=None, num_muestras=10):
    evaluation = []

    for solucion in population:
        total_cost = 0
        total_time = 0
        timespan_ini = 0
        timespan_fin = 0
        acum_timespan_fin = 0
        acum_timespan_ini = 0

        for i, item in enumerate(solucion):
            # Use consumer's own assigned_std if available, else fall back to external array
            if getattr(item.consumer, 'assigned_std', None) is not None:
                std = item.consumer.assigned_std
            elif arrival_time_variance is not None and i < len(arrival_time_variance):
                std = arrival_time_variance[i]
            else:
                std = 0.5  # safe default

            arrival_samples = [random.gauss(item.consumer.available_time_start, std) for _ in range(num_muestras)]
            muestras_salida = [random.gauss(item.consumer.available_time_end, std) for _ in range(num_muestras)]

            sample_costs = 0
            tiempo_muestras = 0

            media_salida = sum(muestras_salida) / len(muestras_salida)

            for muestra_llegada in arrival_samples:
                begin_time = muestra_llegada
                end_time = min(muestra_llegada + item.consumer.get_process_time(), media_salida)
                tiempo_carga = end_time - begin_time

                if tiempo_carga > 0:
                    sample_costs += item.consumer.get_charging_cost(begin_time, end_time, hourly_prices)
                    tiempo_muestras += tiempo_carga

                if i == (len(solucion)-1):
                  acum_timespan_fin += end_time

                if i == 0:
                  acum_timespan_ini = begin_time

            total_cost += (sample_costs / num_muestras)
            total_time += tiempo_muestras / num_muestras
            timespan_fin = acum_timespan_fin / num_muestras
            timespan_ini = acum_timespan_ini / num_muestras

        timespan = timespan_fin - timespan_ini
        evaluation.append((solucion, total_cost, total_time, timespan))

    return evaluation


# FIX Bug 5: Use sorted() instead of sort() to avoid modifying the original list in-place
def parents_selection(evaluation, num_padres=2):
    sorted_evaluation = sorted(evaluation, key=lambda x: (x[1], x[2]))
    padres = []
    for i in range(min(num_padres, len(sorted_evaluation))):
        padres.append(sorted_evaluation[i][0])
    return padres


# FIX Bug 4: Crossover swaps theta_s/theta_f between parents as per paper Algorithm 1:
# crossover((r1,t1,theta_s1,theta_f1),(r2,t2,theta_s2,theta_f2)) = ((r1,t1,theta_s2,theta_f1),(r2,t2,theta_s1,theta_f2))
def parents_crossover(padre1, padre2, resources):
    son = []

    for i in range(len(resources)):
        resources[i].release()

    for i in range(len(padre1)):

        if random.random() < 0.5:
            begin_time_son = padre1[i].begin_time
            end_time_son = padre2[i].end_time
        else:
            begin_time_son = padre2[i].begin_time
            end_time_son = padre1[i].end_time

        if begin_time_son >= end_time_son:
            begin_time_son, end_time_son = end_time_son, begin_time_son
        if begin_time_son >= end_time_son:
            end_time_son = begin_time_son + 0.5

        consumer = padre1[i].consumer
        begin_time_son = max(begin_time_son, consumer.available_time_start)
        end_time_son = min(end_time_son, consumer.available_time_end)
        if begin_time_son >= end_time_son:
            begin_time_son = consumer.available_time_start
            end_time_son = min(consumer.available_time_start + 0.5, consumer.available_time_end)

        cargador = None
        for j in range(len(resources)):
            if(resources[j].is_occupied(begin_time_son, end_time_son) == False):
              cargador = resources[j]
              resources[j].set_state(begin_time_son, end_time_son)
              break

        item = item_planlist(padre1[i].consumer, begin_time_son, end_time_son, cargador)

        if (cargador == None):
              item = item_planlist(padre1[i].consumer, padre1[i].begin_time, padre1[i].end_time, padre1[i].resource)

        son.append(item)

    return son


#Introduce mutations
def mutate_son(son, mutation_rate, resources):

    for i in range(len(resources)):
      resources[i].release()

    for i in range(len(son)):
        if random.uniform(0, 1) < mutation_rate:
            new_begin_time = random.uniform(son[i].consumer.available_time_start, son[i].consumer.available_time_end - 0.5)
            new_end_time = new_begin_time + son[i].consumer.get_process_time()
            new_end_time = min(new_end_time, son[i].consumer.available_time_end)

            if new_end_time == new_begin_time:
                new_end_time = new_begin_time + 0.5

            for j in range(len(resources)):
              if(resources[j].is_occupied(new_begin_time, new_end_time) == False):
                resources[j].set_state(new_begin_time, new_end_time)
                son[i].begin_time = new_begin_time
                son[i].end_time = new_end_time
                son[i].resource = resources[j]
                break

    return son


#---------------------------PRISM Model Generation-----------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

# Helper: scale assigned_std to PRISM discrete time (x10), minimum 1
def _prism_std(consumer):
    return max(int(round(getattr(consumer, 'assigned_std', 0.5) * 10)), 1)


def generate_evaluation_prism_model(solucion, filename="modelo"):
    
    list_model_vehicles = []

    for item in solucion:
        vehicle_data = {
            'id': int(item.consumer.id),
            'start_time': int(item.begin_time*10),
            'end_time': max(int(item.begin_time*10) + 1, int(item.end_time*10)),
            'charger': int(item.resource.id),
            'std': _prism_std(item.consumer)  # per-vehicle std for PRISM uncertainty
        }
        list_model_vehicles.append(vehicle_data)

    # FIX Bug 1: Variables calculated ONCE before the loop (was inside -> NameError if list empty)
    chargers = {v['charger'] for v in list_model_vehicles}
    num_chargers = len(chargers)
    num_vehicles = len(list_model_vehicles)
    charge_rate = 10
    total_hours = 240

    for vehicle in list_model_vehicles:
        print(vehicle)

    model = "dtmc\n\n"

    model += "// Constantes del sistema\n"
    model += "const int N = {0};\n".format(num_chargers)
    model += "const int M = {0};\n".format(num_vehicles)
    model += "const int charge_rate = {0};\n\n".format(charge_rate)
    model += "const int MAX_BAT = 100;\n"

    model += "const int NOT_AVAILABLE = 0;\n"
    model += "const int CHARGING = 1;\n"
    model += "const int CHARGED = 2;\n\n"

    model += "const int FREE = 0;\n"
    model += "const int OCCUPIED = 1;\n"
    model += "const int BROKEN= 2; \n\n"

    model += "// Seguimiento del tiempo\n"
    model += "const int TOTAL_HOURS = {0};\n".format(total_hours)
    model += "global current_time : [0..TOTAL_HOURS] init 0;\n"
    model += "global phase : [0..M] init 0;\n"
    model += "global start_time : [0..TOTAL_HOURS] init TOTAL_HOURS;\n"
    model += "global end_time : [0..TOTAL_HOURS] init 0;\n\n"

    model += "module time\n"
    model += "    passturn_time: bool init false;\n\n"
    model += "    [] (phase=0) & (!passturn_time) & (current_time < TOTAL_HOURS) ->\n"
    model += "        (current_time' = current_time + 1) & (passturn_time' = true);\n\n"
    model += "    [] (passturn_time) -> (phase' = 1) & (passturn_time' = false);\n"
    model += "    [] (phase=0) & (current_time = TOTAL_HOURS) -> (current_time' = current_time);\n"
    model += "endmodule\n\n"

    for phase_index, vehicle in enumerate(list_model_vehicles, start=1):
        vehicle_id = vehicle['id']
        start_time = vehicle['start_time']
        end_time = vehicle['end_time']
        v_std = vehicle['std']  # per-vehicle std

        model += "module vehicle{0}\n".format(vehicle_id)
        model += "    passturn_vehicle{0}: bool init false;\n".format(vehicle_id)
        model += "    start_time_vehicle{0} : [0..TOTAL_HOURS] init TOTAL_HOURS;\n".format(vehicle_id)
        model += "    end_time_vehicle{0} : [0..TOTAL_HOURS] init 0;\n\n".format(vehicle_id)
        model += "    v{0}_update_timespan : bool init false;\n".format(vehicle_id)
        model += "    v{0}_charge_status : [NOT_AVAILABLE..CHARGED] init NOT_AVAILABLE;\n".format(vehicle_id)
        model += "    v{0}_battery : [0..100] init 20;\n".format(vehicle_id)
        model += "    v{0}_start_time : [0..TOTAL_HOURS] init {1};\n".format(vehicle_id, start_time)
        model += "    v{0}_end_time : [0..TOTAL_HOURS] init {1};\n\n".format(vehicle_id, end_time)
        model += "    v{0}_initialization_init_done: bool init false;\n\n".format(vehicle_id)
        model += "    v{0}_initialization_end_done: bool init false;\n\n".format(vehicle_id)

        # Dynamic probabilities - using per-vehicle std
        model += "    [] (phase={1}) & (!passturn_vehicle{0}) & (!v{0}_initialization_init_done) ->\n".format(vehicle_id, phase_index)
        model += "        0.1 : (v{0}_start_time' = max(0, floor({1} - 2 * {2}))) & (v{0}_initialization_init_done' = true) +\n".format(vehicle_id, start_time, v_std)
        model += "        0.2 : (v{0}_start_time' = max(0, floor({1} - {2}))) & (v{0}_initialization_init_done' = true) +\n".format(vehicle_id, start_time, v_std)
        model += "        0.4 : (v{0}_start_time' = {1}) & (v{0}_initialization_init_done' = true) +\n".format(vehicle_id, start_time)
        model += "        0.2 : (v{0}_start_time' = min(TOTAL_HOURS, floor({1} + {2}))) & (v{0}_initialization_init_done' = true) +\n".format(vehicle_id, start_time, v_std)
        model += "        0.1 : (v{0}_start_time' = min(TOTAL_HOURS, floor({1} + 2 * {2}))) & (v{0}_initialization_init_done' = true); \n\n".format(vehicle_id, start_time, v_std)

        model += "    [] (phase={1}) & (!passturn_vehicle{0}) & (v{0}_initialization_init_done) & (!v{0}_initialization_end_done) ->\n".format(vehicle_id, phase_index)
        model += "        0.1 : (v{0}_end_time' = max(0, floor({1} - 2 * {2}))) & (v{0}_initialization_end_done' = true) +\n".format(vehicle_id, end_time, v_std)
        model += "        0.2 : (v{0}_end_time' = max(0, floor({1} - {2}))) & (v{0}_initialization_end_done' = true) +\n".format(vehicle_id, end_time, v_std)
        model += "        0.4 : (v{0}_end_time' = {1}) & (v{0}_initialization_end_done' = true) +\n".format(vehicle_id, end_time)
        model += "        0.2 : (v{0}_end_time' = min(TOTAL_HOURS, floor({1} + {2}))) & (v{0}_initialization_end_done' = true) +\n".format(vehicle_id, end_time, v_std)
        model += "        0.1 : (v{0}_end_time' = min(TOTAL_HOURS, floor({1} + 2 * {2}))) & (v{0}_initialization_end_done' = true); \n\n".format(vehicle_id, end_time, v_std)

        model += "    [] (phase={1}) & (!passturn_vehicle{0}) & (v{0}_initialization_end_done) & (v{0}_initialization_init_done) & (current_time < v{0}_start_time | v{0}_charge_status = CHARGED)->\n".format(vehicle_id, phase_index)
        model += "        (passturn_vehicle{0}' = true);\n\n".format(vehicle_id)

        model += "    [start_charge{0}] (phase={1}) & (!passturn_vehicle{0}) & (v{0}_initialization_end_done) & (v{0}_initialization_init_done) & (current_time = v{0}_start_time & v{0}_charge_status != CHARGING & v{0}_battery < MAX_BAT)->\n".format(vehicle_id, phase_index)
        model += "        (v{0}_charge_status' = CHARGING) & (v{0}_battery' = min(v{0}_battery + charge_rate, MAX_BAT)) & (passturn_vehicle{0}' = true) & ".format(vehicle_id)
        model += "(start_time_vehicle{0}' = min(start_time, current_time));\n\n".format(vehicle_id)
        
        model += "    [] (phase={1}) & (!passturn_vehicle{0}) & (v{0}_initialization_end_done) & (v{0}_initialization_init_done) & (v{0}_charge_status = CHARGING) & (v{0}_battery < MAX_BAT) ".format(vehicle_id, phase_index)
        model += "& (current_time > v{0}_start_time) & (current_time < v{0}_end_time) ->\n".format(vehicle_id)
        model += "        (v{0}_battery' = min(v{0}_battery + charge_rate, 100)) & (passturn_vehicle{0}' = true);\n\n".format(vehicle_id)
        
        model += "    [wait_for_charge{0}] (phase={1}) & (!passturn_vehicle{0}) & (v{0}_initialization_end_done) & (v{0}_initialization_init_done) & (current_time = v{0}_start_time & v{0}_charge_status != CHARGING & v{0}_battery < MAX_BAT) ->\n".format(vehicle_id, phase_index)
        model += "        (v{0}_start_time' = min(v{0}_start_time + 2, v{0}_end_time-1)) & (passturn_vehicle{0}' = true);\n\n".format(vehicle_id)
        model += "    [] (phase={1}) & (!passturn_vehicle{0}) & (v{0}_initialization_end_done) & (v{0}_initialization_init_done) & (current_time = v{0}_start_time) & (v{0}_charge_status != CHARGING) & (v{0}_battery < MAX_BAT) ->\n".format(vehicle_id, phase_index)
        model += "        (v{0}_start_time' = min(v{0}_start_time + 2, v{0}_end_time-1)) & (passturn_vehicle{0}' = true);\n\n".format(vehicle_id)
        
        model += "    [release_charge{0}] (phase={1}) & (!passturn_vehicle{0}) & (v{0}_initialization_end_done) & (v{0}_initialization_init_done) & (v{0}_battery = MAX_BAT) & (v{0}_charge_status = CHARGING) ->\n".format(vehicle_id, phase_index)
        model += "        (v{0}_charge_status' = CHARGED) & (passturn_vehicle{0}' = true) & (end_time_vehicle{0}' = max(end_time, current_time)) & (v{0}_update_timespan' = true);\n\n".format(vehicle_id)

        model += "    [release_charge{0}] (phase={1}) & (!passturn_vehicle{0}) & (v{0}_initialization_end_done) & (v{0}_initialization_init_done) & (current_time >= v{0}_end_time) & (v{0}_battery < MAX_BAT) & (v{0}_charge_status = CHARGING)->\n".format(vehicle_id, phase_index)
        model += "        (v{0}_charge_status' = NOT_AVAILABLE) & (passturn_vehicle{0}' = true) & (v{0}_update_timespan' = true) & (end_time_vehicle{0}' = max(end_time, current_time));\n\n".format(vehicle_id)
        
        model += "    [] (phase={1}) & (!passturn_vehicle{0}) & (v{0}_initialization_end_done) & (v{0}_initialization_init_done) & (v{0}_battery = MAX_BAT) & (v{0}_charge_status != CHARGING) ->\n".format(vehicle_id, phase_index)
        model += "        (passturn_vehicle{0}' = true);\n\n".format(vehicle_id)

        model += "    [] (phase={1}) & (!passturn_vehicle{0}) & (v{0}_initialization_end_done) & (v{0}_initialization_init_done) & (current_time >= v{0}_end_time) & (v{0}_battery < MAX_BAT) & (v{0}_charge_status != CHARGING)->\n ".format(vehicle_id, phase_index)
        model += "        (passturn_vehicle{0}' = true);\n\n ".format(vehicle_id)
        # FIX DEADLOCK: escape for missed window (current_time > start_time, never started charging)
        model += "    [] (phase={1}) & (!passturn_vehicle{0}) & (v{0}_initialization_end_done) & (v{0}_initialization_init_done) & (current_time > v{0}_start_time) & (current_time < v{0}_end_time) & (v{0}_charge_status = NOT_AVAILABLE)->\n".format(vehicle_id, phase_index)
        model += "        (passturn_vehicle{0}' = true);\n\n".format(vehicle_id)

        model += "    [] (passturn_vehicle{0}) -> (phase' = (phase = M ? 0 : phase + 1)) & (passturn_vehicle{0}' = false) & (end_time' = v{0}_update_timespan ? max(end_time, end_time_vehicle{0}): end_time) & (start_time' = min(start_time, start_time_vehicle{0}));\n".format(vehicle_id)
        model += "endmodule\n\n"

    for charger_id in chargers:
        model += "module charger{0}\n".format(charger_id)
        model += "    charger{0}_status : [FREE..BROKEN] init FREE;\n".format(charger_id)
        model += "    repair_time_charger{0} : [0..TOTAL_HOURS] init 0;\n".format(charger_id)
        model += "    num_fallos{0} : [0..5] init 0;\n\n".format(charger_id)
        model += "    [] (phase=0) & (current_time<240) & (num_fallos{0}<5) & (charger{0}_status = FREE)->\n".format(charger_id)
        model += "        0.1 : (charger{0}_status'= BROKEN) +\n".format(charger_id)
        model += "        0.9 : (charger{0}_status'= charger{0}_status) & (repair_time_charger{0}' = current_time + 2);\n\n".format(charger_id)
        model += "    [] (phase=0) & (current_time<240) & (charger{0}_status = OCCUPIED) -> (charger{0}_status' = charger{0}_status);\n".format(charger_id)
        model += "    [] (phase=0) & (current_time<240) & (charger{0}_status = BROKEN) -> (charger{0}_status' = charger{0}_status);\n".format(charger_id)
        model += "    [] (phase=0) & (current_time<240) & (num_fallos{0}>=5) -> (charger{0}_status' = charger{0}_status);\n\n".format(charger_id)

        for vehicle in list_model_vehicles:
            if vehicle['charger'] == charger_id:
                vehicle_id = vehicle['id']
                model += "    [start_charge{0}] (charger{1}_status = FREE) -> (charger{1}_status' = OCCUPIED);\n".format(vehicle_id, charger_id)
                model += "    [release_charge{0}] (charger{1}_status = OCCUPIED) -> (charger{1}_status' = FREE);\n".format(vehicle_id, charger_id)
                model += "    [wait_for_charge{0}] (charger{1}_status = BROKEN) -> (charger{1}_status' = FREE);\n\n".format(vehicle_id, charger_id)

        model += "endmodule\n\n"

    model += "rewards \"total_time\"\n"
    for vehicle in list_model_vehicles:
        model += "    [] phase = 0 & !passturn_time & v{0}_charge_status = CHARGING : 1;\n".format(vehicle['id'])
    model += "endrewards\n\n"

    model += "rewards \"total_cost\"\n"
    for vehicle in list_model_vehicles:
        model += "    [] phase = 0 & !passturn_time & v{0}_charge_status = CHARGING & (current_time >= 0 & current_time < 60) : 2;\n".format(vehicle['id'])
        model += "    [] phase = 0 & !passturn_time & v{0}_charge_status = CHARGING & (current_time >= 60 & current_time < 100) : 5;\n".format(vehicle['id'])
        model += "    [] phase = 0 & !passturn_time & v{0}_charge_status = CHARGING & (current_time >= 100 & current_time < 180) : 8;\n".format(vehicle['id'])
        model += "    [] phase = 0 & !passturn_time & v{0}_charge_status = CHARGING & (current_time >= 180 & current_time < 240) : 6;\n".format(vehicle['id'])
    model += "endrewards\n\n"

    model += "rewards \"charging_timespan\"\n"
    model += "    [] (phase = 0 & !passturn_time & current_time < TOTAL_HOURS & end_time > start_time) : 1;\n"
    model += "endrewards\n"

    try:
        with open(models_dir+'modelo1.prism', 'w') as file:
            file.write(model)
        print("Modelo PRISM guardado en el archivo 'modelo.txt'.")
    except Exception as e:
        print("Error al guardar el archivo: {}".format(e))

    print("Modelo PRISM guardado en el archivo '{0}.txt'.".format(filename))
    return "Modelo generado"


def generate_prism_model(solucion, filename="modelo"):
    
    list_model_vehicles = []

    for item in solucion:
        vehicle_data = {
            'id': int(item.consumer.id),
            'start_time': int(item.begin_time*10),
            'end_time': max(int(item.begin_time*10) + 1, int(item.end_time*10)),
            'charger': int(item.resource.id),
            'std': _prism_std(item.consumer)  # per-vehicle std for PRISM uncertainty
        }
        print("hey")
        list_model_vehicles.append(vehicle_data)

    # FIX Bug 1: Variables calculated ONCE before the loop
    chargers = {v['charger'] for v in list_model_vehicles}
    num_chargers = len(chargers)
    num_vehicles = len(list_model_vehicles)
    charge_rate = 10
    total_hours = 240

    for vehicle in list_model_vehicles:
        print(vehicle)

    model = "dtmc\n\n"

    model += "// Constantes del sistema\n"
    model += "const int N = {0};\n".format(num_chargers)
    model += "const int M = {0};\n".format(num_vehicles)
    model += "const int charge_rate = {0};\n\n".format(charge_rate)
    model += "const int MAX_BAT = 100;\n"

    model += "const int NOT_AVAILABLE = 0;\n"
    model += "const int CHARGING = 1;\n"
    model += "const int CHARGED = 2;\n\n"

    model += "const int FREE = 0;\n"
    model += "const int OCCUPIED = 1; \n\n"

    model += "// Seguimiento del tiempo\n"
    model += "const int TOTAL_HOURS = {0};\n".format(total_hours)
    model += "global current_time : [0..TOTAL_HOURS] init 0;\n"
    model += "global phase : [0..M] init 0;\n"
    model += "global start_time : [0..TOTAL_HOURS] init TOTAL_HOURS;\n"
    model += "global end_time : [0..TOTAL_HOURS] init 0;\n\n"

    model += "module time\n"
    model += "    passturn_time: bool init false;\n\n"
    model += "    [] (phase=0) & (!passturn_time) & (current_time < TOTAL_HOURS) ->\n"
    model += "        (current_time' = current_time + 1) & (passturn_time' = true);\n\n"
    model += "    [] (passturn_time) -> (phase' = 1) & (passturn_time' = false);\n"
    model += "    [] (phase=0) & (current_time = TOTAL_HOURS) -> (current_time' = current_time);\n"
    model += "endmodule\n\n"

    for phase_index, vehicle in enumerate(list_model_vehicles, start=1):
        vehicle_id = vehicle['id']
        start_time = vehicle['start_time']
        end_time = vehicle['end_time']
        v_std = vehicle['std']

        model += "module vehicle{0}\n".format(vehicle_id)
        model += "    passturn_vehicle{0}: bool init false;\n".format(vehicle_id)
        model += "    start_time_vehicle{0} : [0..TOTAL_HOURS] init TOTAL_HOURS;\n".format(vehicle_id)
        model += "    end_time_vehicle{0} : [0..TOTAL_HOURS] init 0;\n\n".format(vehicle_id)
        model += "    v{0}_update_timespan : bool init false;\n".format(vehicle_id)
        model += "    v{0}_charge_status : [NOT_AVAILABLE..CHARGED] init NOT_AVAILABLE;\n".format(vehicle_id)
        model += "    v{0}_battery : [0..100] init 20;\n".format(vehicle_id)
        model += "    v{0}_start_time : [0..TOTAL_HOURS] init {1};\n".format(vehicle_id, start_time)
        model += "    v{0}_end_time : [0..TOTAL_HOURS] init {1};\n\n".format(vehicle_id, end_time)
        
        model += "    [] (phase={1}) & (!passturn_vehicle{0}) & (current_time < v{0}_start_time | v{0}_charge_status = CHARGED)->\n".format(vehicle_id, phase_index)
        model += "        (passturn_vehicle{0}' = true);\n\n".format(vehicle_id)

        model += "    [start_charge{0}] (phase={1}) & (!passturn_vehicle{0}) & (current_time = v{0}_start_time & v{0}_charge_status != CHARGING & v{0}_battery < MAX_BAT)->\n".format(vehicle_id, phase_index)
        model += "        (v{0}_charge_status' = CHARGING) & (v{0}_battery' = min(v{0}_battery + charge_rate, MAX_BAT)) & (passturn_vehicle{0}' = true) & ".format(vehicle_id)
        model += "(start_time_vehicle{0}' = min(start_time, current_time));\n\n".format(vehicle_id)
        
        model += "    [] (phase={1}) & (!passturn_vehicle{0}) & (v{0}_charge_status = CHARGING) & (v{0}_battery < MAX_BAT) ".format(vehicle_id, phase_index)
        model += "& (current_time > v{0}_start_time) & (current_time < v{0}_end_time) ->\n".format(vehicle_id)
        model += "        (v{0}_battery' = min(v{0}_battery + charge_rate, 100)) & (passturn_vehicle{0}' = true);\n\n".format(vehicle_id)
        
        model += "    [release_charge{0}] (phase={1}) & (!passturn_vehicle{0}) & (v{0}_battery = MAX_BAT) & (v{0}_charge_status = CHARGING) ->\n".format(vehicle_id, phase_index)
        model += "        (v{0}_charge_status' = CHARGED) & (passturn_vehicle{0}' = true) & (end_time_vehicle{0}' = max(end_time, current_time)) & (v{0}_update_timespan' = true);\n\n".format(vehicle_id)

        model += "    [release_charge{0}] (phase={1}) & (!passturn_vehicle{0}) & (current_time >= v{0}_end_time) & (v{0}_battery < MAX_BAT) & (v{0}_charge_status = CHARGING)->\n".format(vehicle_id, phase_index)
        model += "        (v{0}_charge_status' = NOT_AVAILABLE) & (passturn_vehicle{0}' = true) & (v{0}_update_timespan' = true) & (end_time_vehicle{0}' = max(end_time, current_time));\n\n".format(vehicle_id)
        
        model += "    [] (phase={1}) & (!passturn_vehicle{0}) & (v{0}_battery = MAX_BAT) & (v{0}_charge_status != CHARGING) ->\n".format(vehicle_id, phase_index)
        model += "        (passturn_vehicle{0}' = true);\n\n".format(vehicle_id)

        model += "    [] (phase={1}) & (!passturn_vehicle{0}) & (current_time >= v{0}_end_time) & (v{0}_battery < MAX_BAT) & (v{0}_charge_status != CHARGING)->\n ".format(vehicle_id, phase_index)
        model += "        (passturn_vehicle{0}' = true);\n\n ".format(vehicle_id)

        # FIX DEADLOCK: escape for missed window (current_time > start_time, never started charging)
        model += "    [] (phase={1}) & (!passturn_vehicle{0}) & (current_time > v{0}_start_time) & (current_time < v{0}_end_time) & (v{0}_charge_status = NOT_AVAILABLE)->\n".format(vehicle_id, phase_index)
        model += "        (passturn_vehicle{0}' = true);\n\n".format(vehicle_id)

        model += "    [] (passturn_vehicle{0}) -> (phase' = (phase = M ? 0 : phase + 1)) & (passturn_vehicle{0}' = false) & (end_time' = v{0}_update_timespan ? max(end_time, end_time_vehicle{0}): end_time) & (start_time' = min(start_time, start_time_vehicle{0}));\n".format(vehicle_id)
        model += "endmodule\n\n"

    for charger_id in chargers:
        model += "module charger{0}\n".format(charger_id)
        model += "    charger{0}_status : [FREE..OCCUPIED] init FREE;\n\n".format(charger_id)

        for vehicle in list_model_vehicles:
            if vehicle['charger'] == charger_id:
                vehicle_id = vehicle['id']
                model += "    [start_charge{0}] (charger{1}_status = FREE) -> (charger{1}_status' = OCCUPIED);\n".format(vehicle_id, charger_id)
                model += "    [release_charge{0}] (charger{1}_status = OCCUPIED) -> (charger{1}_status' = FREE);\n".format(vehicle_id, charger_id)
                model += "    [wait_for_charge{0}] (charger{1}_status = OCCUPIED) -> (charger{1}_status' = charger{1}_status);\n\n".format(vehicle_id, charger_id)

        model += "endmodule\n\n"

    model += "rewards \"total_time\"\n"
    for vehicle in list_model_vehicles:
        model += "    [] phase = 0 & !passturn_time & v{0}_charge_status = CHARGING : 1;\n".format(vehicle['id'])
    model += "endrewards\n\n"

    model += "rewards \"total_cost\"\n"
    for vehicle in list_model_vehicles:
        model += "    [] phase = 0 & !passturn_time & v{0}_charge_status = CHARGING & (current_time >= 0 & current_time < 60) : 2;\n".format(vehicle['id'])
        model += "    [] phase = 0 & !passturn_time & v{0}_charge_status = CHARGING & (current_time >= 60 & current_time < 100) : 5;\n".format(vehicle['id'])
        model += "    [] phase = 0 & !passturn_time & v{0}_charge_status = CHARGING & (current_time >= 100 & current_time < 180) : 8;\n".format(vehicle['id'])
        model += "    [] phase = 0 & !passturn_time & v{0}_charge_status = CHARGING & (current_time >= 180 & current_time < 240) : 6;\n".format(vehicle['id'])
    model += "endrewards\n\n"

    model += "rewards \"charging_timespan\"\n"
    model += "    [] (phase = 0 & !passturn_time & current_time < TOTAL_HOURS & end_time > start_time) : 1;\n"
    model += "endrewards\n"

    try:
        with open(models_dir+'modelito.prism', 'w') as file:
            file.write(model)
        print("Modelo PRISM guardado en el archivo 'modelo.txt'.")
    except Exception as e:
        print("Error al guardar el archivo: {}".format(e))

    print("Modelo PRISM guardado en el archivo '{0}.txt'.".format(filename))
    return "Modelo generado"


#Generate robot model
def generate_evaluation_robot_model(sol):

    list_model_patients = []
    
    for item in sol:
        # FIX Bug 7: Was using item.patient and item.robot (don't exist).
        patient_data = {
            'id': int(item.consumer.id),
            'start_time': int(item.begin_time*10),
            'end_time': max(int(item.begin_time*10) + 1, int(item.end_time*10)),
            'robot': int(item.resource.id),
            'std': _prism_std(item.consumer)  # per-patient std
        }
        list_model_patients.append(patient_data)

    # FIX Bug 1: Variables calculated ONCE before the loop
    robots = {p['robot'] for p in list_model_patients}
    num_robots = len(robots)
    num_patients = len(list_model_patients)
    robot_speed = 10
    total_hours = 240

    for patient in list_model_patients:
        print(patient)

    model = "dtmc\n\n"

    model += "// Constantes del sistema\n"
    model += "const int N = {0};\n".format(num_robots)
    model += "const int M = {0};\n".format(num_patients)
    model += "const int robot_speed = {0};\n\n".format(robot_speed)
    model += "const int DISTANCE = 100;\n"
    model += "const int NOT_AVAILABLE = 0;\n"
    model += "const int FEEDING = 1;\n"
    model += "const int FEEDED = 2;\n\n"
    model += "const int FREE = 0;\n"
    model += "const int BROKEN = 2;\n"
    model += "const int OCCUPIED = 1;\n\n"

    model += "// Seguimiento del tiempo\n"
    model += "const int TOTAL_HOURS = {0};\n".format(total_hours)
    model += "global current_time : [0..TOTAL_HOURS] init 0;\n"
    model += "global phase : [0..M] init 0;\n"
    model += "global start_time : [0..TOTAL_HOURS] init TOTAL_HOURS;\n"
    model += "global end_time : [0..TOTAL_HOURS] init 0;\n\n"

    model += "module time\n"
    model += "    passturn_time: bool init false;\n\n"
    model += "    [] (phase=0) & (!passturn_time) & (current_time < TOTAL_HOURS) ->\n"
    model += "        (current_time' = current_time + 1) & (passturn_time' = true);\n\n"
    model += "    [] (passturn_time) -> (phase' = 1) & (passturn_time' = false);\n"
    model += "    [] (phase=0) & (current_time = TOTAL_HOURS) -> (current_time' = current_time);\n"
    model += "endmodule\n\n"

    for phase_index, patient in enumerate(list_model_patients, start=1):
        patient_id = patient['id']
        start_time = patient['start_time']
        end_time = patient['end_time']
        p_std = patient['std']  # per-patient std

        model += "module patient{0}\n".format(patient_id)
        model += "    passturn_patient{0}: bool init false;\n".format(patient_id)
        model += "    start_time_patient{0} : [0..TOTAL_HOURS] init TOTAL_HOURS;\n".format(patient_id)
        model += "    end_time_patient{0} : [0..TOTAL_HOURS] init 0;\n\n".format(patient_id)
        model += "    p{0}_update_timespan : bool init false;\n".format(patient_id)
        model += "    p{0}_feed_status : [NOT_AVAILABLE..FEEDED] init NOT_AVAILABLE;\n".format(patient_id)
        model += "    p{0}_robot_situation : [0..100] init 20;\n".format(patient_id)
        model += "    p{0}_start_time : [0..TOTAL_HOURS] init {1};\n".format(patient_id, start_time)
        model += "    p{0}_end_time : [0..TOTAL_HOURS] init {1};\n\n".format(patient_id, end_time)
        model += "    p{0}_initialization_init_done: bool init false;\n\n".format(patient_id)
        model += "    p{0}_initialization_end_done: bool init false;\n\n".format(patient_id)

        # Dynamic probabilities - using per-patient std
        model += "    [] (phase={1}) & (!passturn_patient{0}) & (!p{0}_initialization_init_done) ->\n".format(patient_id, phase_index)
        model += "        0.1 : (p{0}_start_time' = max(0, floor({1} - 2 * {2}))) & (p{0}_initialization_init_done' = true) +\n".format(patient_id, start_time, p_std)
        model += "        0.2 : (p{0}_start_time' = max(0, floor({1} - {2}))) & (p{0}_initialization_init_done' = true) +\n".format(patient_id, start_time, p_std)
        model += "        0.4 : (p{0}_start_time' = {1}) & (p{0}_initialization_init_done' = true) +\n".format(patient_id, start_time)
        model += "        0.2 : (p{0}_start_time' = min(TOTAL_HOURS, floor({1} + {2}))) & (p{0}_initialization_init_done' = true) +\n".format(patient_id, start_time, p_std)
        model += "        0.1 : (p{0}_start_time' = min(TOTAL_HOURS, floor({1} + 2 * {2}))) & (p{0}_initialization_init_done' = true); \n\n".format(patient_id, start_time, p_std)

        model += "    [] (phase={1}) & (!passturn_patient{0}) & (p{0}_initialization_init_done) & (!p{0}_initialization_end_done) ->\n".format(patient_id, phase_index)
        model += "        0.1 : (p{0}_end_time' = max(0, floor({1} - 2 * {2}))) & (p{0}_initialization_end_done' = true) +\n".format(patient_id, end_time, p_std)
        model += "        0.2 : (p{0}_end_time' = max(0, floor({1} - {2}))) & (p{0}_initialization_end_done' = true) +\n".format(patient_id, end_time, p_std)
        model += "        0.4 : (p{0}_end_time' = {1}) & (p{0}_initialization_end_done' = true) +\n".format(patient_id, end_time)
        model += "        0.2 : (p{0}_end_time' = min(TOTAL_HOURS, floor({1} + {2}))) & (p{0}_initialization_end_done' = true) +\n".format(patient_id, end_time, p_std)
        model += "        0.1 : (p{0}_end_time' = min(TOTAL_HOURS, floor({1} + 2 * {2}))) & (p{0}_initialization_end_done' = true); \n\n".format(patient_id, end_time, p_std)

        model += "    [] (phase={1}) & (!passturn_patient{0}) & (p{0}_initialization_end_done) & (p{0}_initialization_init_done) & (current_time < p{0}_start_time | p{0}_feed_status = FEEDED)->\n".format(patient_id, phase_index)
        model += "        (passturn_patient{0}' = true);\n\n".format(patient_id)
        model += "    [start_feed{0}] (phase={1}) & (!passturn_patient{0}) & (p{0}_initialization_end_done) & (p{0}_initialization_init_done) & (current_time = p{0}_start_time & p{0}_feed_status != FEEDING & p{0}_robot_situation < DISTANCE)->\n".format(patient_id, phase_index)
        model += "        (p{0}_feed_status' = FEEDING) & (p{0}_robot_situation' = min(p{0}_robot_situation + robot_speed, DISTANCE)) & (passturn_patient{0}' = true) & (start_time_patient{0}' = min(start_time, current_time));\n\n".format(patient_id)
        model += "    [] (phase={1}) & (!passturn_patient{0}) & (p{0}_initialization_end_done) & (p{0}_initialization_init_done) & (p{0}_feed_status = FEEDING) & (p{0}_robot_situation < DISTANCE) & (current_time > p{0}_start_time) & (current_time < p{0}_end_time) ->\n".format(patient_id, phase_index)
        model += "        (p{0}_robot_situation' = min(p{0}_robot_situation + robot_speed, 100)) & (passturn_patient{0}' = true);\n\n".format(patient_id)
        model += "    [release_feed{0}] (phase={1}) & (!passturn_patient{0}) & (p{0}_initialization_end_done) & (p{0}_initialization_init_done) & (p{0}_robot_situation = DISTANCE) & (p{0}_feed_status = FEEDING) ->\n".format(patient_id, phase_index)
        model += "        (p{0}_feed_status' = FEEDED) & (passturn_patient{0}' = true) & (p{0}_update_timespan' = true);\n\n".format(patient_id)
        model += "    [release_feed{0}] (phase={1}) & (!passturn_patient{0}) & (p{0}_initialization_end_done) & (p{0}_initialization_init_done) & (current_time >= p{0}_end_time) & (p{0}_robot_situation < DISTANCE) & (p{0}_feed_status = FEEDING)->\n".format(patient_id, phase_index)
        model += "        (p{0}_feed_status' = NOT_AVAILABLE) & (passturn_patient{0}' = true) & (end_time_patient{0}' = max(end_time, current_time));\n\n".format(patient_id)
        model += "    [] (phase={1}) & (!passturn_patient{0}) & (p{0}_initialization_end_done) & (p{0}_initialization_init_done) & (p{0}_robot_situation = DISTANCE) & (p{0}_feed_status != FEEDING) ->\n".format(patient_id, phase_index)
        model += "        (passturn_patient{0}' = true);\n\n".format(patient_id)
        model += "    [] (phase={1}) & (!passturn_patient{0}) & (p{0}_initialization_end_done) & (p{0}_initialization_init_done) & (current_time >= p{0}_end_time) & (p{0}_robot_situation < DISTANCE) & (p{0}_feed_status != FEEDING)->\n ".format(patient_id, phase_index)
        model += "        (passturn_patient{0}' = true);\n\n ".format(patient_id)
        # FIX DEADLOCK: escape for missed window (current_time > start_time, never started feeding)
        model += "    [] (phase={1}) & (!passturn_patient{0}) & (p{0}_initialization_end_done) & (p{0}_initialization_init_done) & (current_time > p{0}_start_time) & (current_time < p{0}_end_time) & (p{0}_feed_status = NOT_AVAILABLE)->\n".format(patient_id, phase_index)
        model += "        (passturn_patient{0}' = true);\n\n".format(patient_id)
        model += "    [] (passturn_patient{0}) -> (phase' = (phase = M ? 0 : phase + 1)) & (passturn_patient{0}' = false) & (end_time' = p{0}_update_timespan ? max(end_time, end_time_patient{0}): end_time) & (start_time' = min(start_time, start_time_patient{0}));\n".format(patient_id)
        model += "endmodule\n\n"

    for robot_id in robots:
        model += "module robot{0}\n".format(robot_id)
        model += "    robot{0}_status : [FREE..BROKEN] init FREE;\n\n".format(robot_id)
        model += "    repair_time_robot{0} : [0..TOTAL_HOURS] init 0;\n".format(robot_id)
        model += "    num_fallos{0} : [0..5] init 0;\n\n".format(robot_id)
        model += "    [] (phase=0) & (current_time<240) & (num_fallos{0}<5) & (robot{0}_status = FREE)->\n".format(robot_id)
        model += "        0.1 : (robot{0}_status'= BROKEN) +\n".format(robot_id)
        model += "        0.9 : (robot{0}_status'= robot{0}_status) & (repair_time_robot{0}' = current_time + 2);\n\n".format(robot_id)
        model += "    [] (phase=0) & (current_time<240) & (robot{0}_status = OCCUPIED) -> (robot{0}_status' = robot{0}_status);\n".format(robot_id)
        model += "    [] (phase=0) & (current_time<240) & (robot{0}_status = BROKEN) -> (robot{0}_status' = robot{0}_status);\n".format(robot_id)
        model += "    [] (phase=0) & (current_time<240) & (num_fallos{0}>=5) -> (robot{0}_status' = robot{0}_status);\n\n".format(robot_id)

        for patient in list_model_patients:
            if patient['robot'] == robot_id:
                patient_id = patient['id']
                model += "    [start_feed{0}] (robot{1}_status = FREE) -> (robot{1}_status' = OCCUPIED);\n".format(patient_id, robot_id)
                model += "    [release_feed{0}] (robot{1}_status = OCCUPIED) -> (robot{1}_status' = FREE);\n\n".format(patient_id, robot_id)
                model += "    [wait_for_feed{0}] (robot{1}_status = BROKEN) -> (robot{1}_status' = FREE);\n\n".format(patient_id, robot_id)

        model += "endmodule\n\n"

    model += "rewards \"total_time\"\n"
    for patient in list_model_patients:
        model += "    [] phase = 0 & !passturn_time & p{0}_feed_status = FEEDING : 1;\n".format(patient['id'])
    model += "endrewards\n\n"

    model += "rewards \"total_cost\"\n"
    for patient in list_model_patients:
        model += "    [] phase = 0 & !passturn_time & p{0}_feed_status = FEEDING & (current_time >= 0 & current_time < 60) : 2;\n".format(patient['id'])
        model += "    [] phase = 0 & !passturn_time & p{0}_feed_status = FEEDING & (current_time >= 60 & current_time < 120) : 4;\n".format(patient['id'])
        model += "    [] phase = 0 & !passturn_time & p{0}_feed_status = FEEDING & (current_time >= 120) : 6;\n".format(patient['id'])
    model += "endrewards\n\n"
    
    model += "rewards \"charging_timespan\"\n"
    model += "    [] (phase = 0 & !passturn_time & current_time < TOTAL_HOURS & end_time > start_time) : 1;\n"
    model += "endrewards\n\n"
    
    model += "rewards \"total_disruption\"\n"
    for patient in list_model_patients:
        model += "    [] phase = 0 & !passturn_time & p{0}_feed_status = FEEDING & (current_time >= p{0}_start_time & current_time < p{0}_end_time) : 1;\n".format(patient['id'])
    model += "endrewards\n"

    try:
        with open(models_dir+'modelorobots2.prism', 'w') as file:
            file.write(model)
        print("Modelo PRISM guardado en el archivo 'modelorobots2.prism'.")
    except Exception as e:
        print("Error al guardar el archivo: {}".format(e))


#Method to select one of the scenarios
def select_generate_model(solucion):
    try:
        with open('C:/Users/raque/OneDrive/Escritorio/fichero.txt', 'r') as file:
            primera_linea = file.readline().strip()
            partes = primera_linea.split(',')
            if len(partes) >= 2 and partes[1].isdigit():
                caso_estudio = partes[0].lower()
                if caso_estudio in ["robots"]:
                    generate_evaluation_robot_model(solucion)
                elif caso_estudio in ["vehiculos"]:
                    generate_evaluation_prism_model(solucion)
                else:
                    print("No coincide con 'robots' ni 'vehiculos'")
            else:
                print("Formato incorrecto en la línea")
    except IOError:
        print("El archivo no se encontró.")
    except Exception as e:
        print("Ocurrió un error: {}".format(e))


#Final generic method to generate the PRISM model from JSON config
def generate_evaluation_model_final(sol, config):
    list_model = []

    for item in sol:
        entity_data = {
            'id': int(item.consumer.id),
            'start_time': int(item.begin_time * 10),
            'end_time': max(int(item.begin_time * 10) + 1, int(item.end_time * 10)),
            'resource': int(item.resource.id),
            'std': _prism_std(item.consumer)  # per-entity std for PRISM uncertainty
        }
        list_model.append(entity_data)

    print("\n".join(map(str, list_model)))

    resources = {e['resource'] for e in list_model}
    num_resources = len(resources)
    num_entities = len(list_model)

    speed_value = config["speed_value"]
    total_hours = 240
    max_param_value = config["max_value"]

    model = "dtmc\n\n"

    model += "// Constantes del sistema\n"
    model += "const int N = {0};\n".format(num_resources)
    model += "const int M = {0};\n".format(num_entities)
    model += "const int {0} = {1};\n\n".format(config['speed_param'], speed_value)
    model += "const int {0} = {1};\n\n".format(config['max_param'], max_param_value)
    model += "const int {0} = 0;\n".format(config['state_available'])
    model += "const int {0} = 1;\n".format(config['state_active'])
    model += "const int {0} = 2;\n\n".format(config['state_done'])
    model += "const int FREE = 0;\nconst int OCCUPIED = 1;\nconst int BROKEN = 2;\n\n"

    model += "// Seguimiento del tiempo\n"
    model += "const int TOTAL_HOURS = {0};\n".format(total_hours)
    model += "global current_time : [0..TOTAL_HOURS] init 0;\n"
    model += "global phase : [0..M] init 0;\n"
    model += "global start_time : [0..TOTAL_HOURS] init TOTAL_HOURS;\n"
    model += "global end_time : [0..TOTAL_HOURS] init 0;\n\n"
    
    model += "module time\n"
    model += "    passturn_time: bool init false;\n\n"
    model += "    [] (phase=0) & (!passturn_time) & (current_time < TOTAL_HOURS) ->\n"
    model += "        (current_time' = current_time + 1) & (passturn_time' = true);\n\n"
    model += "    [] (passturn_time) -> (phase' = 1) & (passturn_time' = false);\n"
    model += "    [] (phase=0) & (current_time = TOTAL_HOURS) -> (current_time' = current_time);\n"
    model += "endmodule\n\n"

    for phase_index, entity in enumerate(list_model, start=1):
        entity_id = entity['id']
        start_time = entity['start_time']
        end_time = entity['end_time']
        e_std = entity['std']  # per-entity std

        model += "module {0}{1}\n".format(config['entity'], entity_id)
        model += "    passturn_{0}{1}: bool init false;\n".format(config['entity'], entity_id)
        model += "    start_time_{0}{1} : [0..TOTAL_HOURS] init TOTAL_HOURS;\n".format(config['entity'],entity_id)
        model += "    end_time_{0}{1} : [0..TOTAL_HOURS] init 0;\n\n".format(config['entity'],entity_id)
        model += "    {0}{1}_update_timespan : bool init false;\n".format(config['entity'][0],entity_id)
        model += "    {0}{1}_{4}_status : [{2}..{3}] init {2};\n".format(config['entity'][0], entity_id, config['state_available'], config['state_done'], config['action'])
        model += "    {0}{1}_{3}_situation : [0..{2}] init 20;\n".format(config['entity'][0], entity_id, max_param_value, config['progress'])
        model += "    {0}{1}_start_time : [0..TOTAL_HOURS] init {2};\n".format(config['entity'][0], entity_id, start_time)
        model += "    {0}{1}_end_time : [0..TOTAL_HOURS] init {2};\n\n".format(config['entity'][0], entity_id, end_time)
        model += "    {0}{1}_initialization_init_done: bool init false;\n\n".format(config['entity'][0],entity_id)
        model += "    {0}{1}_initialization_end_done: bool init false;\n\n".format(config['entity'][0],entity_id)
        
        # Dynamic probabilities - using per-entity std
        model += "    [] (phase={2}) & (!passturn_{0}{3}) & (!{1}{3}_initialization_init_done) ->\n".format(config['entity'], config['entity'][0], phase_index, entity_id)
        model += "        0.1 : ({3}{0}_start_time' = max(0, floor({1} - 2 * {2}))) & ({3}{0}_initialization_init_done' = true) +\n".format(entity_id, start_time, e_std, config['entity'][0])
        model += "        0.2 : ({3}{0}_start_time' = max(0, floor({1} - {2}))) & ({3}{0}_initialization_init_done' = true) +\n".format(entity_id, start_time, e_std, config['entity'][0])
        model += "        0.4 : ({2}{0}_start_time' = {1}) & ({2}{0}_initialization_init_done' = true) +\n".format(entity_id, start_time, config['entity'][0])
        model += "        0.2 : ({3}{0}_start_time' = min(TOTAL_HOURS, floor({1} + {2}))) & ({3}{0}_initialization_init_done' = true) +\n".format(entity_id, start_time, e_std, config['entity'][0])
        model += "        0.1 : ({3}{0}_start_time' = min(TOTAL_HOURS, floor({1} + 2 * {2}))) & ({3}{0}_initialization_init_done' = true); \n\n".format(entity_id, start_time, e_std, config['entity'][0])

        model += "    [] (phase={2}) & (!passturn_{0}{3}) & ({1}{3}_initialization_init_done) & (!{1}{3}_initialization_end_done) ->\n".format(config['entity'], config['entity'][0], phase_index, entity_id)
        model += "        0.1 : ({3}{0}_end_time' = max(0, floor({1} - 2 * {2}))) & ({3}{0}_initialization_end_done' = true) +\n".format(entity_id, end_time, e_std, config['entity'][0])
        model += "        0.2 : ({3}{0}_end_time' = max(0, floor({1} - {2}))) & ({3}{0}_initialization_end_done' = true) +\n".format(entity_id, end_time, e_std, config['entity'][0])
        model += "        0.4 : ({2}{0}_end_time' = {1}) & ({2}{0}_initialization_end_done' = true) +\n".format(entity_id, end_time, config['entity'][0])
        model += "        0.2 : ({3}{0}_end_time' = min(TOTAL_HOURS, floor({1} + {2}))) & ({3}{0}_initialization_end_done' = true) +\n".format(entity_id, end_time, e_std, config['entity'][0])
        model += "        0.1 : ({3}{0}_end_time' = min(TOTAL_HOURS, floor({1} + 2 * {2}))) & ({3}{0}_initialization_end_done' = true); \n\n".format(entity_id, end_time, e_std, config['entity'][0])

        model += "    [] (phase={5}) & (!passturn_{4}{0}) & ({1}{0}_initialization_end_done) & ({1}{0}_initialization_init_done) & (current_time < {1}{0}_start_time | {1}{0}_{2}_status = {3})->\n".format(entity_id, config['entity'][0], config['action'], config['state_done'], config['entity'], phase_index)
        model += "        (passturn_{1}{0}' = true);\n\n".format(entity_id, config['entity'])
        model += "    [{1}{0}] (phase={9}) & (!passturn_{2}{0}) & ({3}{0}_initialization_end_done) & ({3}{0}_initialization_init_done) & (current_time = {3}{0}_start_time & {3}{0}_{4}_status != {5} & {3}{0}_{8}_situation < {7})->\n".format(entity_id, config['start_action'], config['entity'], config['entity'][0], config['action'], config['state_active'], config['resource'], config['max_param'], config['progress'], phase_index)
        model += "        ({1}{0}_{3}_status' = {4}) & ({1}{0}_{8}_situation' = min({1}{0}_{8}_situation + {6}, {5})) & (passturn_{2}{0}' = true) & (start_time_{2}{0}' = min(start_time, current_time));\n\n".format(entity_id, config['entity'][0], config['entity'], config['action'], config['state_active'], config['max_param'], config['speed_param'], config['resource'], config['progress'])
        model += "    [] (phase={9}) & (!passturn_{4}{0}) & ({1}{0}_initialization_end_done) & ({1}{0}_initialization_init_done) & ({1}{0}_{2}_status = {3}) & ({1}{0}_{8}_situation < {7}) & (current_time > {1}{0}_start_time) & (current_time < {1}{0}_end_time) ->\n".format(entity_id, config['entity'][0], config['action'], config['state_active'], config['entity'], config['resource'], config['resource'], config['max_param'], config['progress'], phase_index)
        model += "        ({1}{0}_{8}_situation' = min({1}{0}_{8}_situation + {5}, {7})) & (passturn_{2}{0}' = true);\n\n".format(entity_id, config['entity'][0], config['entity'], config['action'], config['state_active'], config['speed_param'], config['resource'], config['max_param'], config['progress'])
        model += "    [wait_for_{1}{0}] (phase={7}) & (!passturn_{2}{0}) & ({3}{0}_initialization_end_done) & ({3}{0}_initialization_init_done) & (current_time = {3}{0}_start_time & {3}{0}_{1}_status != {4} & {3}{0}_{5}_situation < {6}) ->\n".format(entity_id, config['action'], config['entity'], config['entity'][0], config['state_active'], config['progress'], config['max_param'], phase_index)
        model += "        ({2}{0}_start_time' = min({2}{0}_start_time + 2, {2}{0}_end_time-1)) & (passturn_{1}{0}' = true);\n\n".format(entity_id, config['entity'], config['entity'][0])
        model += "    [] (phase={7}) & (!passturn_{2}{0}) & ({3}{0}_initialization_end_done) & ({3}{0}_initialization_init_done) & (current_time = {3}{0}_start_time) & ({3}{0}_{1}_status != {4}) & ({3}{0}_{5}_situation < {6}) ->\n".format(entity_id, config['action'], config['entity'], config['entity'][0], config['state_active'], config['progress'], config['max_param'], phase_index)
        model += "        ({2}{0}_start_time' = min({2}{0}_start_time + 2, {2}{0}_end_time-1)) & (passturn_{1}{0}' = true);\n\n".format(entity_id, config['entity'], config['entity'][0])
        model += "    [{1}{0}] (phase={9}) & (!passturn_{4}{0}) & ({2}{0}_initialization_end_done) & ({2}{0}_initialization_init_done) & ({2}{0}_{8}_situation = {6}) & ({2}{0}_{3}_status = {7}) ->\n".format(entity_id, config['release_action'], config['entity'][0], config['action'], config['entity'], config['resource'], config['max_param'], config['state_active'], config['progress'], phase_index)
        model += "        ({2}{0}_{3}_status' = {8}) & (passturn_{4}{0}' = true) & (end_time_{4}{0}' = max(end_time, current_time)) & ({2}{0}_update_timespan' = true);\n\n".format(entity_id, config['release_action'], config['entity'][0], config['action'], config['entity'], config['state_active'], config['resource'], config['max_param'], config['state_done'])
        model += "    [{1}{0}] (phase={9}) & (!passturn_{4}{0}) & ({2}{0}_initialization_end_done) & ({2}{0}_initialization_init_done) & (current_time >= {2}{0}_end_time) & ({2}{0}_{8}_situation < {6}) & ({2}{0}_{3}_status = {7}) ->\n".format(entity_id, config['release_action'], config['entity'][0], config['action'], config['entity'], config['resource'], config['max_param'], config['state_active'], config['progress'], phase_index)
        model += "        ({2}{0}_{3}_status' = {8}) & (passturn_{4}{0}' = true) & ({2}{0}_update_timespan' = true) & (end_time_{4}{0}' = max(end_time, current_time));\n\n".format(entity_id, config['release_action'], config['entity'][0], config['action'], config['entity'], config['state_active'], config['resource'], config['max_param'], config['state_available'])
        model += "    [] (phase={9}) & (!passturn_{4}{0}) & ({2}{0}_initialization_end_done) & ({2}{0}_initialization_init_done) & ({2}{0}_{8}_situation = {6}) & ({2}{0}_{3}_status != {7}) ->\n".format(entity_id, config['entity'][0], config['entity'][0], config['action'], config['entity'], config['resource'], config['max_param'], config['state_active'], config['progress'], phase_index)
        model += "        (passturn_{4}{0}' = true);\n\n".format(entity_id, config['entity'][0], config['entity'], config['action'], config['entity'])
        model += "    [] (phase={9}) & (!passturn_{4}{0}) & ({2}{0}_initialization_end_done) & ({2}{0}_initialization_init_done) & (current_time >= {2}{0}_end_time) & ({2}{0}_{8}_situation < {6}) & ({2}{0}_{3}_status != {7}) ->\n ".format(entity_id, config['entity'][0], config['entity'][0], config['action'], config['entity'], config['resource'], config['max_param'], config['state_active'], config['progress'], phase_index)
        model += "        (passturn_{4}{0}' = true);\n\n ".format(entity_id, config['entity'][0], config['entity'], config['action'], config['entity'])
        model += "    [] (passturn_{4}{0}) -> (phase' = (phase = M ? 0 : phase + 1)) & (passturn_{4}{0}' = false) & (end_time' = {1}{0}_update_timespan ? max(end_time, end_time_{4}{0}): end_time) & (start_time' = min(start_time, start_time_{4}{0}));\n".format(entity_id, config['entity'][0], config['entity'], config['action'], config['entity'])
        model += "endmodule\n\n"

    for resource_id in resources:
        model += "module {0}{1}\n".format(config['resource'], resource_id)
        model += "    {0}{1}_status : [FREE..BROKEN] init FREE;\n".format(config['resource'], resource_id)
        model += "    repair_time_{0}{1} : [0..TOTAL_HOURS] init 0;\n".format(config['resource'], resource_id)
        model += "    num_fallos{0} : [0..5] init 0;\n\n".format(resource_id)
        model += "    [] (phase=0) & (current_time<TOTAL_HOURS) & (num_fallos{0}<5) & ({1}{0}_status = FREE)->\n".format(resource_id, config['resource'])
        model += "        0.1 : ({1}{0}_status'= BROKEN) +\n".format(resource_id, config['resource'])
        model += "        0.9 : ({1}{0}_status'= {1}{0}_status) & (repair_time_{1}{0}' = current_time + 2);\n\n".format(resource_id, config['resource'])
        model += "    [] (phase=0) & (current_time<TOTAL_HOURS) & ({1}{0}_status = OCCUPIED) -> ({1}{0}_status' = {1}{0}_status);\n".format(resource_id, config['resource'])
        model += "    [] (phase=0) & (current_time<TOTAL_HOURS) & ({1}{0}_status = BROKEN) -> ({1}{0}_status' = {1}{0}_status);\n".format(resource_id, config['resource'])
        model += "    [] (phase=0) & (current_time<TOTAL_HOURS) & (num_fallos{0}>=5) -> ({1}{0}_status' = {1}{0}_status);\n\n".format(resource_id, config['resource'])

        for entity in list_model:
            if entity['resource'] == resource_id:
                current_entity_id = entity['id']
                model += "    [{0}{1}] ({2}{3}_status = FREE) -> ({2}{3}_status' = OCCUPIED);\n".format(config['start_action'], current_entity_id, config['resource'], resource_id)
                model += "    [{0}{1}] ({2}{3}_status = OCCUPIED) -> ({2}{3}_status' = FREE);\n".format(config['release_action'], current_entity_id, config['resource'], resource_id)
                model += "    [wait_for_{2}{0}] ({3}{1}_status = BROKEN) -> ({3}{1}_status' = FREE);\n\n".format(current_entity_id, resource_id, config['action'], config['resource'])

        model += "endmodule\n\n"

    model += "rewards \"total_time\"\n"
    for entity in list_model:
        model += "    [] phase = 0 & !passturn_time & {1}{0}_{2}_status = {3} : 1;\n".format(entity['id'], config['entity'][0], config['action'], config['state_active'])
    model += "endrewards\n\n"

    if config['reward_acum'] == "true":
        model += "rewards \"total_cost\"\n"
        for entity in list_model:
            model += "    [] phase = 0 & !passturn_time & {1}{0}_{2}_status = {3} & (current_time >= 0 & current_time < 60) : 2;\n".format(entity['id'], config['entity'][0], config['action'], config['state_active'])
            model += "    [] phase = 0 & !passturn_time & {1}{0}_{2}_status = {3} & (current_time >= 60 & current_time < 120) : 4;\n".format(entity['id'], config['entity'][0], config['action'], config['state_active'])
            model += "    [] phase = 0 & !passturn_time & {1}{0}_{2}_status = {3} & (current_time >= 120) : 6;\n".format(entity['id'], config['entity'][0], config['action'], config['state_active'])
        model += "endrewards\n\n"

    if config['reward_acum_const'] == "true":
        model += "rewards \"total_cost\"\n"
        for entity in list_model:
            model += "    [] phase = 0 & !passturn_time & {1}{0}_{2}_status = {3}: 100;\n".format(entity['id'], config['entity'][0], config['action'], config['state_active'])
        model += "endrewards\n\n"

    if config['reward_timespan'] == "true":
        model += "rewards \"{0}_timespan\"\n".format(config['action'])
        model += "    [] (phase = 0 & !passturn_time & current_time < TOTAL_HOURS & end_time > start_time) : 1;\n"
        model += "endrewards\n\n"

    model += "rewards \"total_disruption\"\n"
    for entity in list_model:
        model += "    [] phase = 0 & !passturn_time & {1}{0}_{2}_status = {3} & (current_time >= {1}{0}_start_time & current_time < {1}{0}_end_time) : 1;\n".format(entity['id'], config['entity'][0], config['action'], config['state_active'])
    model += "endrewards\n\n"

    # FIX Bug 6: Was using entity_id from the last iteration of the previous loop.
    # Now uses current_entity_id correctly for each entity.
    model += "rewards \"fully_completed\"\n"
    for entity in list_model:
        current_entity_id = entity['id']
        model += "    [{1}{0}] ({2}{0}_{8}_situation = {6}) & ({2}{0}_{3}_status = {7}) : 1;\n".format(
            current_entity_id, config['release_action'], config['entity'][0], config['action'], config['entity'], config['resource'], config['max_param'], config['state_active'], config['progress']
        )
    model += "endrewards\n"

    try:
        with open(models_dir+'modelomodelohey.prism', 'w') as file:
            file.write(model)
        print("Modelo PRISM guardado en el archivo 'modelomodelohey.prism'.")
    except Exception as e:
        print("Error al guardar el archivo: {}".format(e))


import json

#Select configuration file depending on the scenario
def generate_evaluation_model_config(sol):
    config = {}
    with open(config_dir+"configuracion_vehiculos.json", "r") as file:
        config = json.load(file)
    print(config)
    generate_evaluation_model_final(sol, config)
