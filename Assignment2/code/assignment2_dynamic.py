#%%
import numpy as np
import pandas as pd
from math import pi, sqrt, sin, cos, ceil
from debugger import debug

# Read the excel files
airports = pd.read_excel("AirportData.xlsx", index_col=0)
fleet_options = pd.read_excel("FleetType.xlsx", index_col=0)
demand_df = pd.read_excel("Group35.xlsx", skiprows=range(1,5), index_col=0)

#number of airports
AP = len(airports.columns)

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
demand = np.zeros((AP, AP,30))
for i in range(AP):
    for j in range(AP):
        for k in range(30):
            df_index = i * 20 + j + 1
            demand[i,j,k] = demand_df.loc[df_index, str(k)]
# For example the demand from Airport 0 (LHR) to Airport 1 (CDG) for the 5th time slot
total_demand = np.copy(demand)
radian = pi/180
airports = pd.read_excel("AirportData.xlsx", index_col=0)
runways = airports.loc[["Runway (m)"]].to_numpy().flatten()

def d(pos0: list, pos1: list):
    R = 6371
    #from the assignment:
    return 2*R*sqrt(sin((pos0[0]-pos1[0])/2)**2 + cos(pos0[0])*cos(pos1[0])*sin((pos0[1]-pos1[1])/2)**2)

n = len(airports.axes[1])
#Create n x n distance matrix where n is the number of airports considered.
dist = np.zeros([n,n])
for i in range(n):
    for j in range(n):
        poss = airports.loc[["Latitude (deg)", "Longitude (deg)"]].astype(float)
        loc0 = poss.iloc[:,i]
        loc1 = poss.iloc[:,j]
        dist[i,j] = d(loc0.to_numpy()*radian, loc1.to_numpy()*radian)

#fuel price in USD/gal
fuel_price = 1.42
yield_coeff = 0.26

#%%
aircraft_types = pd.read_excel("FleetType.xlsx", index_col=0)

#types[:,0] gives characteristics of aircraft type 1
types = aircraft_types.to_numpy()

#%%
#index 3 is FRA, our hub
hub = 3

class Aircraft:
    def __init__(self, ac_type):
        self.speed, self.capacity, self.tat, self.range,\
        self.runway_length, self.lease_cost, self.fixed_cost,\
        self.hour_cost, self.fuel_cost, _ = types.T[ac_type]

fleet = types[-1]


#%%
#function f definition
@debug
def f(ac: Aircraft, location, time, cargo, dest, demand):
    #numpy array must be copied in recursion
    demand = np.copy(demand)

    if time>1200:
        print("time's up")
        return -np.inf, None, None, None
    
    if time == 1200:
        print("1200")
        if location != hub:
            return -np.inf, None, None, None
        return 0, [location], [time], demand

    if dest == location:
        tt=1

    profit=0
    #if flight does not visit hub
    if (location == hub) == (dest == hub) == False:
        return -np.inf, None, None, None
    else:
        #we gaan vliegen!!!!!
        d = dist[location, hub]
        if d > ac.range or runways[dest] < ac.runway_length:
            #return infeasible if runway or range do not match.
            return -np.inf, None, None, None
        
        flight_hours = d/ac.speed

        cost = ac.fixed_cost + ac.hour_cost*flight_hours + ac.fuel_cost*fuel_price*d/1.5

        #uitladen @location (niet dest)
        cargo[location] = 0

        #inladen @location (niet dest)

        #hoe laat is het
        timeslot = time//40

        #First, take all possible cargo from 2 time slots ago
        if timeslot>1:
            load = min(demand[location, dest, timeslot-2], 0.2*total_demand[location, dest, timeslot-2], ac.capacity-cargo.sum())
            cargo[dest] += load
            demand[location, dest, timeslot-2] -= load

        #Take all possible cargo from previous time slot
        if timeslot>0:
            load = min(demand[location, dest, timeslot-1], 0.2*total_demand[location, dest, timeslot-1], ac.capacity - cargo.sum())
            cargo[dest] += load
            demand[location, dest, timeslot-1] -= load

        #take all possible cargo from current time slot
        load = min(demand[location, dest, timeslot], ac.capacity-cargo.sum())
        cargo[dest] += load
        demand[location, dest, timeslot] -= load

        revenue = yield_coeff*d*cargo.sum()

        profit = revenue - cost

        #calc travel time. add 2*15 minutes takeoff & landing time, scale to model timescale.
        #round up
        tt = ceil((flight_hours+0.5)*10)

        #preparing for next step
        location = dest
    
    outcomes = [f(ac, location, time+tt, cargo, dest, demand) for dest in range(AP)]

    p, route, times, demand_res =\
        max(outcomes, key = lambda x: x[0])
    try:
        profit_res = p + profit
        route_res = [dest] + route
        times_res = [time] + times
    except TypeError as foutje:
        #raise Exception(foutje, locals()["outcomes"])
        print(locals()["outcomes"])
    return profit_res, route_res, times_res, demand_res

#demand d_ijt demand in timeframe t from airport i to airport j.

#%%
stop = False
while not stop:
    max_profit = -np.inf
    opt_route = []
    opt_times = []

    for ac_type in range(3):
        if fleet[ac_type] > 0:
            ac = Aircraft(ac_type)
            location = hub
            time = 0
            cargo = np.zeros(AP)
            profit, route, times, demand_res =\
                max([f(ac, location, time, cargo, dest, demand) for dest in range(AP)],\
                                                   key = lambda x: x[0])

            if profit > max_profit:
                max_profit = profit
                opt_route = route
                opt_times = times
                opt_demand = demand_res
                opt_ac_type = ac_type

    if np.any(fleet == 0) or max_profit < 0:
        stop = True
    else:
        demand = opt_demand
        fleet[opt_ac_type] -=1


