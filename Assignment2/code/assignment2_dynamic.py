#%%
import numpy as np
import pandas as pd
from gurobipy import Model, GRB, quicksum
import time

"""# Step 1: Load data from GroupX.xlsx
file = "Group_35.xlsx"
flights = pd.read_excel(file, sheet_name="Flights",index_col=0)  # Flight schedule
itineraries = pd.read_excel(file, sheet_name="Itineraries")  # Passenger itineraries
recapture = pd.read_excel(file, sheet_name="Recapture")  # Recapture rates
num_flights = range(len(flights))
num_itineraries = range(len(itineraries))
num_recaptures = range(len(recapture))
"""
aircraft_types = pd.read_excel("FleetType.xlsx", index_col=0)

#types[:,0] gives characteristics of aircraft type 1
types = aircraft_types.to_numpy()

#%%

#index 3 is FRA, our hub
hub = 3
def f(type, demand, location = hub):
    profit, route, cargo = max([f(type, demand, location, choice) for choice in range(20)], key = lambda x: x[0])
    return profit, route, cargo 


fleet = types[-1]

stop = False
while not stop:
    max_profit = -np.inf

    for type in range(3):
        if fleet(type) > 0:

            profit = f(type, demand)



# %%
