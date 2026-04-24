# -*- coding: utf-8 -*-
from __future__ import print_function
import json

config_dir="C:\\Users\\raque\\OneDrive\\Escritorio\\prueba\\general_opt_system\\config_files\\"
general_dir='C:/Users/raque/OneDrive/Escritorio/prueba/general_opt_system/'

# Base classes code compatible with Python 2.
# Consumer now accepts assigned_std so each instance carries its own uncertainty level,
# which is read from the 'assigned_std' column in the CSV data file.
base_classes_code = '''
# -*- coding: utf-8 -*-
from __future__ import print_function

import random

class Consumer:
    def __init__(self, id, distance, available_time_start, available_time_end, assigned_std=0.5, **kwargs):
        self.id = id
        self.distance = distance
        self.available_time_start = available_time_start
        self.available_time_end = available_time_end
        # assigned_std: uncertainty level for this consumer (punctual -> low std, non-punctual -> high std)
        self.assigned_std = assigned_std

    def get_process_time(self):
        raise NotImplementedError("This method should be implemented by subclasses")

class Resource:
    def __init__(self, id):
        self.id = id
        self.occupied_periods = []

    def set_state(self, ini_hour, final_hour):
        self.occupied_periods.append((ini_hour, final_hour))

    def delete_state(self, ini_hour, final_hour):
        self.occupied_periods = [
            (ini, fin) for ini, fin in self.occupied_periods
            if (ini, fin) != (ini_hour, final_hour)
        ]

    def release(self):
        self.occupied_periods = []

    def is_occupied(self, ini_hour, final_hour):
        for inicio, fin in self.occupied_periods:
            if (ini_hour >= inicio and ini_hour <= fin) or (final_hour >= inicio and final_hour <= fin):
                return True
        return False

    def has_occupied_hours(self):
        return bool(self.occupied_periods)
'''


def generate_class_code(class_info):
    name = class_info["name"]
    base = class_info["base_class"]
    attributes = class_info["attributes"]
    methods = class_info.get("methods", [])

    # For Consumer subclasses, include assigned_std and **kwargs so extra CSV columns are ignored
    if base == "Consumer":
        init_params = ", ".join(["self"] + attributes + ["assigned_std=0.5", "**kwargs"])
        init_body = ""
        for attr in attributes:
            init_body += "        self.%s = %s\n" % (attr, attr)
        # Call super().__init__ passing assigned_std
        init_body += "        Consumer.__init__(self, id, distance, available_time_start, available_time_end, assigned_std=assigned_std)\n"
    else:
        init_params = ", ".join(["self"] + attributes)
        init_body = ""
        for attr in attributes:
            init_body += "        self.%s = %s\n" % (attr, attr)

    init_code = "    def __init__(%s):\n%s" % (init_params, init_body)

    method_code = ""
    for method in methods:
        if method == "get_attending_time":
            method_code += '''
    def get_process_time(self):
        if not self.has_food:
            return 2 * (self.distance / 5.0)
        return 0
    '''
        elif method == "get_charging_time":
            method_code += '''
    def get_process_time(self):
        required_charge = self.distance * (self.discharge_rate / 100.0)
        if self.current_charge < required_charge:
            required_charge -= self.current_charge
            return required_charge / self.charge_speed
        else:
            return 0
    '''
        elif method == "get_charging_cost":
            method_code += '''
    def get_charging_cost(self, begin_time, end_time, hourly_prices):
        total_cost = 0
        current_time = begin_time
        while current_time < end_time:
            total_cost += self.charge_speed * hourly_prices[int(current_time)]
            current_time += 1
        return total_cost
    '''
        elif method == "get_irrigation_cost":
            method_code += '''
    def get_irrigation_cost(self, begin_time, end_time, water_price, irrigation_speed):
        total_cost = water_price * irrigation_speed * (end_time - begin_time)
        return total_cost
    '''

    return "\nclass %s(%s):\n%s%s\n" % (name, base, init_code, method_code)


def main():
    with open(config_dir+"configuracion_vehiculos.json", "r") as f:
        config = json.load(f)

    output = base_classes_code
    for cls in config["classes"]:
        output += generate_class_code(cls)

    with open(general_dir+'generated_classes.py', "w") as f:
        f.write(output)

if __name__ == "__main__":
    main()

#TODO: Recordar que en el caso de la agricultura, los cultivos van a taner id, tini, tfin, distancia al robot, nivel de humedad (gramos agua por metro cúbico)
#De momento, para esta primera version se establece, al igual que para el caso de los pacientes, que el tiempo se mida
#segun lo que tarden los robots en llegar a las parcelas y se establezca un tiempo comun para lo que tardan en regarse estas
#O un tiempo que dependa de la humedad del suelo y de lo que necesiten (considerar tiempo que tardan en llegar igual para todos los cultivos)