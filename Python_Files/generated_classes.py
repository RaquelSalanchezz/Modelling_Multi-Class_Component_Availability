
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

class ElectricVehicle(Consumer):
    def __init__(self, id, distance, available_time_start, available_time_end, current_charge, battery_capacity, charge_speed, discharge_rate, assigned_std=0.5, **kwargs):
        self.id = id
        self.distance = distance
        self.available_time_start = available_time_start
        self.available_time_end = available_time_end
        self.current_charge = current_charge
        self.battery_capacity = battery_capacity
        self.charge_speed = charge_speed
        self.discharge_rate = discharge_rate
        Consumer.__init__(self, id, distance, available_time_start, available_time_end, assigned_std=assigned_std)

    def get_process_time(self):
        required_charge = self.distance * (self.discharge_rate / 100.0)
        if self.current_charge < required_charge:
            required_charge -= self.current_charge
            return required_charge / self.charge_speed
        else:
            return 0
    
    def get_charging_cost(self, begin_time, end_time, hourly_prices):
        total_cost = 0
        current_time = begin_time
        while current_time < end_time:
            total_cost += self.charge_speed * hourly_prices[int(current_time)]
            current_time += 1
        return total_cost
    

class Charger(Resource):
    def __init__(self, id):
        self.id = id

