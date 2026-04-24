# -*- coding: utf-8 -*-
from __future__ import division, print_function

import random
import csv
import time

from generated_classes import Charger
from generated_classes import ElectricVehicle
import clasesAG


# Parameters of the genetic algorithm
population_size = 10
num_generations = 50
num_parents = 3
mutation_rate = 0.2

# Generate chargers list
resources = [Charger(1), Charger(2), Charger(3), Charger(4), Charger(5), Charger(6), Charger(7), Charger(8)]


# FIX Bug 9: Removed redundant create_vehicles_from_file() / vehicles2 (.txt method).
# Unified data loading using create_objects_from_csv() which reads assigned_std from the CSV.
def create_objects_from_csv(file_path, cls, field_mapping=None):
    """
    Creates a list of objects from a CSV file.
    Reads assigned_std per row so each consumer has its own uncertainty level.
    Compatible with Python 2 (uses 'rb' and iteritems).
    """
    objects = []

    with open(file_path, 'rb') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if field_mapping:
                mapped_row = {field_mapping.get(k, k): v for k, v in row.iteritems()}
            else:
                mapped_row = dict(row)

            # Convert numeric values
            for k, v in mapped_row.iteritems():
                try:
                    if '.' in str(v):
                        mapped_row[k] = float(v)
                    else:
                        mapped_row[k] = int(v)
                except (ValueError, TypeError):
                    pass  # keep as string if not convertible

            obj = cls(**mapped_row)
            objects.append(obj)

    return objects

#-------------------------------------------------------------------
# Path of the data file (CSV with assigned_std column)
#-------------------------------------------------------------------

file_name = 'C:\\Users\\raque\\OneDrive\\Escritorio\\prueba\\general_opt_system\\vehicles_case\\vehicles_data\\vehicles_clustered_kmeans.csv'
vehicles = create_objects_from_csv(file_name, ElectricVehicle)
print(vehicles)


# Hourly prices
hourly_prices = [0.1, 0.2, 0.15, 0.25, 0.2, 0.15, 0.1, 0.1, 0.2, 0.3, 0.35, 0.3,
                 0.25, 0.2, 0.15, 0.1, 0.1, 0.2, 0.3, 0.35, 0.3, 0.25, 0.2, 0.15]

start_time = time.time()

# Generate initial population
population = clasesAG.generate_initial_population(vehicles, resources, population_size)

# Run genetic algorithm
for generation in range(num_generations):
    # evaluate_fitness uses each consumer's assigned_std automatically
    evaluations = clasesAG.evaluate_fitness(population, hourly_prices, None, None)
    parents = clasesAG.parents_selection(evaluations, num_parents)

    nueva_population = []
    for i in range(population_size):
        parent1 = random.choice(parents)
        parent2 = random.choice(parents)
        son = clasesAG.parents_crossover(parent1, parent2, resources)
        son_mutated = clasesAG.mutate_son(son, mutation_rate, resources)
        nueva_population.append(son_mutated)

    population = nueva_population

# Get the best solution found
evaluations = clasesAG.evaluate_fitness(population, hourly_prices, None, None)
best_evaluations = sorted(evaluations, key=lambda x: (x[1], x[2]))
best_sol = best_evaluations[0]
end_time = time.time()

print("#########################################################################")
print("Total cost: " + str(best_sol[1]))
print("Total time: " + str(best_sol[2]))
print("Timespan: " + str(best_sol[3]))
print("Plan:")

execution_time = end_time - start_time
print("Tiempo total: " + str(execution_time))

# Generate PRISM model
clasesAG.generate_evaluation_model_config(best_sol[0])