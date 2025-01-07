#%%
import numpy as np
import pandas as pd

# Read the excel files
airports = pd.read_excel("AirportData.xlsx", index_col=0)
fleet_options = pd.read_excel("FleetType.xlsx", index_col=0)
demand_df = pd.read_excel("Group35.xlsx", skiprows=range(1,5), index_col=0)

#%%
# Simplify column names for better indexing
fleet_columns = [col.split(':')[0] for col in fleet_options.columns]
fleet_options.columns = fleet_columns
demand_columns = ["Origin", "Destination"] + [str(i) for i in range(30)]
demand_df.columns = demand_columns

# examples of how to access data
#print(Airports.loc['Latitude (deg)', 'London'])
#print(FleetOptions.loc["Speed [km/h]", 'Type 1'])
#print(Demand_df.loc[3,str(5)])

# Restructure the demand data into a 3D array
demand = np.zeros((len(airports.columns), len(airports.columns),30))
for i in range(len(airports.columns)):
    for j in range(len(airports.columns)):
        for k in range(30):
            df_index = i * 20 + j + 1
            demand[i,j,k] = demand_df.loc[df_index, str(k)]
# For example the demand from Airport 0 (LHR) to Airport 1 (CDG) for the 5th time slot
#print(Demand[0,1,5])
#%%
aircraft_types = pd.read_excel("FleetType.xlsx", index_col=0)

#types[:,0] gives characteristics of aircraft type 1
types = aircraft_types.to_numpy()

#%%

#index 3 is FRA, our hub
hub = 3
def f(type, demand, location = hub):
    profit, route, cargo = max([f(type, demand, location, choice) for choice in range(20)], key = lambda x: x[0])
    return profit, route, cargo 

#demand d_tij demand in timeframe t from airport i to airport j.



fleet = types[-1]

stop = False
while not stop:
    max_profit = -np.inf

    for type in range(3):
        if fleet(type) > 0:

            profit = f(type, demand)



# %%
