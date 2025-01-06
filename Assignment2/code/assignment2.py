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
# Simplify column names for cleaner code
new_columns = [col.split(':')[0] for col in FleetOptions.columns]
FleetOptions.columns = new_columns
Demand_df = pd.read_excel(Demand_path, skiprows=1:3 index_col=0)
#print(Airports.loc['Latitude (deg)', 'London'])
#print(FleetOptions.loc["Speed [km/h]", 'Type 1'])
print(Demand_df.columns)
Demand = np.zeros((len(Airports.columns), len(Airports.columns),30))
for i in range(len(Airports.columns)):
    for j in range(len(Airports.columns)):
        for k in range(30):
            Demand[i,j,k] = Demand_df
