#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from math import pi, sqrt, sin, cos, ceil
from debugger import debug
import sys
np.set_printoptions(threshold=sys.maxsize//2)

# Read the excel files
airports = pd.read_excel("AirportData.xlsx", index_col=0)
aircraft_types = pd.read_excel("FleetType.xlsx", index_col=0)
demand_df = pd.read_excel("Group35.xlsx", skiprows=range(1,5), index_col=(0,1,2))

#number of airports
AP = len(airports.columns)
#number of time slots (4h) -2 leading columns
time_slots = demand_df.shape[1]-2
#number of timesteps (in steps of 0.1h = 6m
time_steps = time_slots*40

#calculate great arc length distance between airports
def d(pos0: list, pos1: list):
    R = 6371
    #from the assignment:
    return 2*R*sqrt(sin((pos0[0]-pos1[0])/2)**2 + cos(pos0[0])*cos(pos1[0])*sin((pos0[1]-pos1[1])/2)**2)
radian = pi/180
dist = np.zeros((AP,AP))
for i in range(AP):
    for j in range(AP):
        poss = airports.loc[["Latitude (deg)", "Longitude (deg)"]].astype(float)
        loc0 = poss.iloc[:,i]
        loc1 = poss.iloc[:,j]
        dist[i,j] = d(loc0.to_numpy()*radian, loc1.to_numpy()*radian)

#save runway lengths in numpy array.
runways = airports.loc[["Runway (m)"]].to_numpy().flatten()

#aircraft types
types = aircraft_types.to_numpy()
fleet = aircraft_types.loc[["Fleet"]].to_numpy(dtype=int).flatten()

# Restructure the demand data into a 3D array
#demand_kij is the demand from airport i to airport j in timeslot k.
demand = demand_df.to_numpy().T.reshape(-1, 20, 20)
total_demand = demand.copy()

#parameters
fuel_price = 1.42
yield_coeff = 0.26
hub = 3
min_block = 60 #in 0.1h (6m)

#%%
class Aircraft:
    def __init__(self, ac_type):
        self.speed, self.capacity, self.tat, self.range,\
        self.runway_length, self.lease_cost, self.fixed_cost,\
        self.hour_cost, self.fuel_cost, _ = types.T[ac_type]

        self.ac_type = ac_type

class Node:
    """
    The Node class represents a node in the network of airports.
    When the optimal policy given a certain node is evaluated,
    all followup airports are compared.
    The next node that is visited is based on blocking time.
    The potential destination nodes' parameters are used to calculate
    profit, which is then used to find the best destination.
    The destination's parameters are copied and modified, then added to the current node.
    """
    def __init__(self, airpt, time, block):
        self.airpt = airpt
        self.time = time
        self.block = block #total block time so far (part of system state)
        self.time_slot = time//40
        self.demand = None
        self.cargo = np.zeros(AP)
        self.route = []
        self.times = []
        self.cargos = []
        self.profit = -np.inf
        self.is_hub = airpt == hub
        self.next: Node = None


    def profit(self, network, ac: Aircraft, dest_airpt):
        infeasible = (-np.inf, (None,))
        status = "ok"
        dest: Node

        if self.airpt == dest_airpt:
            pass
            #TODO: no profit, no cargo, ready_time = 1, block_time = 0
            profit = 0
            dest = network[dest_airpt][self.block][self.time+1]
        elif (not self.is_hub) and (dest_airpt != hub):
            return infeasible, "flight does not visit hub"
        else:
            if runways[dest_airpt] < ac.runway_length:
                return infeasible, "infeasible runway length"
            
            #we gaan vliegen
            d = dist[self.airpt, dest_airpt]
            if d > ac.range:
                return infeasible, "infeasible range"
            
            #flight time in hours
            flight_hours = d/ac.speed
            #flight time in 0.1h (6m)
            flight_time = 10*flight_hours

            #add 30m to find block time
            block_time = flight_time + 5
            ready_time = self.time + ceil(block_time + ac.tat/6)
            

            if ready_time > time_steps:
                return infeasible, "time is up"
            
            
            #at the end of 24h, check if the block time limit is satisfied
            #If current time and ready time are not in the same 24h block:
            if self.time//240 != ready_time//240:
                #self.block should be reset to 0. All other cases are infeasible.
                if self.block != 0:
                    return infeasible, "reset block time"
                #choose the node that satisfies dest_airpt and ready time and maximizes profit.
                dest = max([network[dest_airpt][b][ready_time] for b in range(min_block, 240)], key = lambda x: x.profit)
            else:
                #subtract block time, since we are going in reverse
                block_dest = self.block - int(block_time)
                if block_dest < 0:
                    return infeasible, "block time infeasible"
                dest = network[dest_airpt][block_dest][ready_time]

            #now the Node dest has been defined, only use dest.airpt instead of dest_airpt

            if dest.profit == -np.inf:
                return infeasible, "dest not discovered."

            #copy dest (total)demand in order to modify.
            demand = dest.demand.copy()
            profit = 0
            cargo = np.zeros(AP)

            if self.time_slot > 3:
                #load factors of previous time slots
                factors = [1, 0.2, 0.2]
                for j in range(3):
                    load = min(demand[self.time_slot-j, self.airpt, dest.airpt],\
                            ac.capacity - cargo.sum(),\
                                factors[j]*total_demand[self.time_slot-j, self.airpt, dest.airpt])
                    cargo[dest.airpt] += load
                    demand[self.time_slot-j, self.airpt, dest.airpt] -= load

                for j in range(3):
                    load = min(demand[self.time_slot-j, self.airpt, dest.next.airpt],\
                               ac.capacity - max(cargo.sum(), dest.cargo.sum()),\
                               factors[j]*total_demand[self.time_slot-j, self.airpt, dest.next.airpt])
                    cargo[dest.next.airpt] += load
                    dest.cargo[dest.next.airpt] += load
                    demand[self.time_slot-j, self.airpt, dest.next.airpt] -= load
                    profit += dest.update_profit()

            revenue = yield_coeff*d*cargo.sum()/1000

            cost = ac.fixed_cost + ac.hour_cost*flight_hours + ac.fuel_cost*fuel_price*d/1.5

            profit += revenue - cost

            route = [self.airpt] + dest.route
            times = [self.time] + dest.times
            cargos = [cargo] + dest.cargos

            profit += dest.profit
        return profit, ((route, times, cargo), demand), status
        #TODO time slot based demand???? moeilijk
# %%

