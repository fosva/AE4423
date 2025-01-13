#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from math import pi, sqrt, sin, cos, ceil
from debugger import debug

# Read the excel files
airports = pd.read_excel("AirportData.xlsx", index_col=0)
fleet_options = pd.read_excel("FleetType.xlsx", index_col=0)
demand_df = pd.read_excel("Group35.xlsx", skiprows=range(1,5), index_col=0)

#number of airports
AP = len(airports.columns)
#number of time steps (120h divided in 10 steps per hour)
time_steps = 1200
#30 time slots taking 4h each.
time_slots = 30

fleet_columns = [col.split(':')[0] for col in fleet_options.columns]
fleet_options.columns = fleet_columns
demand_columns = ["Origin", "Destination"] + [str(i) for i in range(30)]
demand_df.columns = demand_columns

# Restructure the demand data into a 3D array
#demand_ijk is the demand from airport i to airport j in timeslot k.
demand = np.zeros((AP, AP, time_slots))
for i in range(AP):
    for j in range(AP):
        for k in range(30):
            df_index = i * 20 + j + 1
            demand[i,j,k] = demand_df.loc[df_index, str(k)]

#Demand is given per route and connection of cargo at hub is already considered,
# so we only need to take into account cargo directly to and from hub.
#demand_hub_ijk denotes (i=0 outgoing, i=1 incoming) demand at hub relative to airport j, at timeslot k.
#accesed as demand_hub[dest==hub, j, k]
demand_hub = np.array([demand[0], demand[:,0]])

#%%

total_demand = np.copy(demand)

logbins = np.logspace(0, 6, 70)
demand_nonzero = total_demand[total_demand.nonzero()].flatten()
plt.hist(demand_nonzero, bins = logbins)
plt.title('demand (log scale)')
plt.xscale('log')
#plt.show()
#%%
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

class Node:
    """
    The Node class represents a node in the network of airports. When the optimal policy given a certain node is evaluated,
    all followup nodes are compared, in the comparison. The potential destination nodes' parameters are used to calculate
    profit, which is then used to find the best destination. The destination's parameters are copied and modified, then added to the current node.
    """
    def __init__(self, airpt, time):
        """sd"""
        self.airpt = airpt
        self.time = time
        self.timeslot = time//40
        self.demand = []
        self.route = []
        self.times = []
        self.profit = 0
        self.blocking_time = 0
        self.is_hub = airpt == hub


class Aircraft:
    def __init__(self, ac_type):
        self.speed, self.capacity, self.tat, self.range,\
        self.runway_length, self.lease_cost, self.fixed_cost,\
        self.hour_cost, self.fuel_cost, _ = types.T[ac_type]
        self.ac_type = ac_type

fleet = types[-1]


#%%

#function f definition
#@debug
def f(ac: Aircraft, origin_id, time, dest_id, network):
    infeasible = (-np.inf, None, None, None, None)

    profit=0
    #if aircraft stays at the same spot
    if dest_id == origin_id:
        tt=1

    #aircraft does not visit hub
    elif (origin_id == hub) == (dest_id == hub) == False:
        #print("Flight does not visit hub")
        return infeasible
    else:
        #we gaan vliegen!!!!!
        d = dist[origin_id, dest_id]
        if d > ac.range or runways[dest_id] < ac.runway_length:
            #return infeasible if runway or range do not match.
            #print("Range or runway constraint not met")
            return infeasible
        
        #in 1h
        flight_hours = d/ac.speed
        #in 1h
        blocking_time = flight_hours + 0.5
        #in 0.1h (6m)
        ready_time = time + ceil(10*(blocking_time + ac.tat))
        
        if ready_time >= time_steps:
            return infeasible
        if ready_time == time_steps-1:
            if dest_id != hub:
                return infeasible
        #Get origin and dest from network
        #Node objects
        origin: Node = network[origin_id, time]
        dest: Node = network[dest_id, ready_time]

        cost = ac.fixed_cost + ac.hour_cost*flight_hours + ac.fuel_cost*fuel_price*d/1.5

        #We choose to go to the destination dest. The optimal route from that point on has a demand table associated with it. 
        # This is the one we make calculations with, and then store in the origin Node.
        demand = dest.demand.copy()
        timeslot = origin.timeslot

        cargo = 0
        #Firstly, take all possible cargo from current time slot
        load = min(demand[origin_id, dest_id, timeslot], ac.capacity-cargo)
        cargo += load
        demand[origin_id, dest_id, timeslot] -= load

        #Take all possible cargo from previous time slot
        if timeslot>0:
            load = min(demand[origin_id, dest_id, timeslot-1], ac.capacity - cargo, 0.2*total_demand[origin_id, dest_id, timeslot-1])
            cargo += load
            demand[origin_id, dest_id, timeslot-1] -= load

        #Take all possible cargo from 2 time slots ago
        if timeslot>1:
            load = min(demand[origin_id, dest_id, timeslot-2], ac.capacity-cargo, 0.2*total_demand[origin_id, dest_id, timeslot-2])
            cargo += load
            demand[origin_id, dest_id, timeslot-2] -= load

        revenue = yield_coeff*d*cargo

        profit = revenue - cost
    #TODO figure out blocking time calculation.
    #calc minimum block time requirement: 6 hours per day.
    #time from ending
    #blocking time is measured in 1h steps, time in 0.1h steps (6m).
    inv_time = time_slots - time
    if 6*(inv_time//240) > blocking_time + dest.blocking_time:
        return infeasible
    return profit+dest.profit, [time] + dest.times, [dest.airpt] + dest.route, demand, blocking_time + dest.blocking_time

#%%


ac_types_res = []
result = []
tot_demand = demand_hub.copy()

tot_times = []
tot_routes = []
tot_profit = []

stop = False
while not stop:
    opt_profit = -np.inf
    opt_times = []
    opt_route = []
    opt_blocking_time = 0
    opt_demand = []
    opt_ac_type = -1


    for ac_type in range(3):
        if fleet[ac_type] > 0:
            ac = Aircraft(ac_type)

            #network_airpt,time gives the node connected to the given airport at the given time step (6m)
            network = [[Node(i, j) for j in range(time_steps)] for i in range(AP)]
            #set network demand at hub at time = 1200, based on demand found in previous iteration.
            network[hub, -1].demand = tot_demand.copy()

            #Loop over nodes backwards in time
            for time in range(time_steps-1, -1, -1):
                for origin_id in range(AP):

                    origin: Node = network[origin_id, time]
                    #Given time and origin, find most profitable destination.
                    origin.profit, origin.times, origin.route, origin.demand, origin.blocking_time =\
                        max([f(ac, origin_id, time, dest, network) for dest in range(AP)], key = lambda x: x[0])
            #check if profit found at starting node exceeds previously found profit.
            start_node: Node = network[hub,0]
            if start_node.profit > opt_profit:
                opt_demand = start_node.demand.copy()
                opt_profit = start_node.profit
                opt_ac_type = ac_type


    if np.all(fleet == 0) or opt_profit < 0:
        stop = True
    else:
        #update all totals found
        tot_demand = opt_demand
        fleet[opt_ac_type] -=1
        ac_types_res.append(opt_ac_type)
        result.append([opt_times, opt_route])
        tot_times.append(opt_times)
        tot_routes.append(opt_route)
        tot_profit.append(opt_profit)

print(ac_types_res, result)

#%%
