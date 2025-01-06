import pandas as pd
import numpy as np
import os

# import input data: Airports, fleet and demand data
# First get the path of the datafiles
script_dir = os.path.dirname(os.path.abspath(__file__))
Airports_path = os.path.join(script_dir, '..', 'AirportData.xlsx')
FleetOptions_path = os.path.join(script_dir, '..', 'FleetType.xlsx')
Demand_path = os.path.join(script_dir, '..', 'Group35.xlsx')

# Read the excel files
Airports = pd.read_excel(Airports_path, index_col=0)
FleetOptions = pd.read_excel(FleetOptions_path, index_col=0)
Demand_df = pd.read_excel(Demand_path, skiprows=range(1,5), index_col=0)

# Simplify column names for better indexing
fleet_columns = [col.split(':')[0] for col in FleetOptions.columns]
FleetOptions.columns = fleet_columns
demand_columns = ["Origin", "Destination"] + [str(i) for i in range(30)]
Demand_df.columns = demand_columns

# examples of how to access data
#print(Airports.loc['Latitude (deg)', 'London'])
#print(FleetOptions.loc["Speed [km/h]", 'Type 1'])
#print(Demand_df.loc[3,str(5)])

# Restructure the demand data into a 3D array
Demand = np.zeros((len(Airports.columns), len(Airports.columns),30))
for i in range(len(Airports.columns)):
    for j in range(len(Airports.columns)):
        for k in range(30):
            df_index = i * 20 + j + 1
            Demand[i,j,k] = Demand_df.loc[df_index, str(k)]
# For example the demand from Airport 0 (LHR) to Airport 1 (CDG) for the 5th time slot
#print(Demand[0,1,5])