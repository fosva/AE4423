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

# number of airports
AP = len(airports.columns)
#number of time slots (4h) -2 leading columns
time_slots = demand_df.shape[1]
#number of timesteps (in steps of 0.1h = 6m
time_steps = time_slots*40

# calculate great arc length distance between airports and store them in an array
def d(pos0: list, pos1: list):
    R = 6371
    # from the assignment:
    return 2*R*sqrt(sin((pos0[0]-pos1[0])/2)**2 + cos(pos0[0])*cos(pos1[0])*sin((pos0[1]-pos1[1])/2)**2)
radian = pi/180
dist = np.zeros((AP,AP))
for i in range(AP):
    for j in range(AP):
        poss = airports.loc[["Latitude (deg)", "Longitude (deg)"]].astype(float)
        loc0 = poss.iloc[:,i]
        loc1 = poss.iloc[:,j]
        dist[i,j] = d(loc0.to_numpy()*radian, loc1.to_numpy()*radian)

# save runway lengths in numpy array.
runways = airports.loc[["Runway (m)"]].to_numpy().flatten()

# aircraft types
types = aircraft_types.to_numpy()
fleet = aircraft_types.loc[["Fleet"]].to_numpy(dtype=int).flatten()

# Restructure the demand data into a 3D array
# demand_kij is the demand from airport i to airport j in timeslot k.
demand = demand_df.to_numpy().T.reshape(-1, 20, 20)
total_demand = demand.copy()

# parameters
fuel_price = 1.42
yield_coeff = 0.26
hub = 3
min_block = 60 #in 0.1h (6m)

#%%
# Aircraft class, stores aircraft parameters for every type
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
    def __init__(self, airpt, time):
        self.airpt = airpt # airport that the node
        self.time = time  #timestep of the node
        self.block_time = 0 #total block time so far
        self.time_slot = time//40
        self.demand = None # demand at this airport and timestep, updated at every node update
        self.route = [airpt] # list of airports of optimal route from this airport for the remaining timesteps
        self.times = [time] # list of timesteps of optimal route from this airport for the remaining timesteps
        self.cargos = np.zeros((1,AP)) # the cargos loaded to the aircraft when arriving at this airport
        self.profit = -np.inf  # profit of optimal route from this airport for the remaining timesteps, initialized at -inf
        self.is_hub = airpt == hub # whether the airport is the hub or not
        self.next: Node = None # next node in the optimal route (i.e. node associated with destination airport at the timstep of arrival)

    # Function to calculate profit of a flight from this node to a specified destination with a specified aircraft type
    def calc_profit(self, network, ac: Aircraft, dest_airpt):
        # Initializations
        infeasible = (-np.inf, (None,))
        status = "ok"
        dest: Node
        
        profit = 0
        cargo = np.zeros(AP)  # will be updated with the cargo loaded to this flight sorted for destination airport
        cargo2 = np.zeros(AP) # will be updated with the cargo loaded to next flight.
        block_time = 0

        # if destination airport is the same as the current airport, destination is the node of the same airport at the next timestep, block time and demand are unchanged
        if self.airpt == dest_airpt:
            #aircraft stays on ground
            if self.time ==  1199:
                print(self.time, self.airpt, dest_airpt, ac.ac_type)
            dest = network[dest_airpt][self.time+1]
            if dest.profit == -np.inf:
                return infeasible, "dest not discovered _"+str(dest)
            demand = dest.demand.copy()
            cargos = [np.zeros(AP)] + dest.cargos

        # if neither origin nor destination is the hub, the flight is infeasible
        elif (not self.is_hub) and (dest_airpt != hub):
            return infeasible, "flight does not visit hub"

        else:
            #we gaan vliegen
            # check runway and range constraints, return infeasible if not satisfied
            if runways[dest_airpt] < ac.runway_length:
                return infeasible, "infeasible runway length"
            
            d = dist[self.airpt, dest_airpt]
            if d > ac.range:
                return infeasible, "infeasible range"
            
            # compute travel time to update block time and the time of arrival at the destination
            #flight time in hours
            flight_hours = d/ac.speed
            # flight time in 0.1h (6m), or timesteps
            flight_time = 10*flight_hours

            # add 30m to calc block time
            block_time = flight_time + 5
            ready_time = self.time + ceil(block_time + ac.tat/6)

            # If the time of arrival is beyond the scheduling horizon, the flight is infeasible
            if ready_time > time_steps-1:
                return infeasible, "time is up"
            dest: Node = network[dest_airpt][ready_time]
            #now the Node dest has been defined, only use dest.airpt instead of dest_airpt

            #now the Node dest has been defined, only use dest.airpt instead of dest_airpt
            if dest.profit == -np.inf:
                return infeasible, "dest not discovered."
            demand = dest.demand.copy()

            # Determine the cargo that can be loaded at the origin airport including 20% of the 2 previous time slots if capacity allows
            dest_cargo = dest.cargos[0].copy()
            if self.time_slot > 3:
                #load factors of previous time slots
                factors = [1, 0.2, 0.2]
                for j in range(3):
                    load = min(demand[self.time_slot-j, self.airpt, dest.airpt],\
                            ac.capacity - cargo.sum(),\
                                factors[j]*total_demand[self.time_slot-j, self.airpt, dest.airpt])
                    cargo[dest.airpt] += load # store in cargo array
                    demand[self.time_slot-j, self.airpt, dest.airpt] -= load # update demand by substracting the loaded cargo

                # if possible, load cargo destined for the airport after the destination airport
                if dest.next is not None:

                    for j in range(3):
                        load = min(demand[self.time_slot-j, self.airpt, dest.next.airpt],\
                                ac.capacity - max(cargo.sum(), dest_cargo.sum()),\
                                factors[j]*total_demand[self.time_slot-j, self.airpt, dest.next.airpt])
                        dest_cargo[dest.next.airpt] += load # store separately for destination node
                        cargo2[dest.next.airpt] += load
                        demand[self.time_slot-j, self.airpt, dest.next.airpt] -= load # update demand by substracting the loaded cargo



            # Calculate revenue based on cargo loaded
            revenue = yield_coeff*d*(cargo + cargo2).sum()/1000

            # Calculate cost based on aircraft type and flight distance
            cost = ac.fixed_cost + ac.hour_cost*flight_hours + ac.fuel_cost*fuel_price*d/1.5

            profit += revenue - cost

            #add cargo used in next leg to current leg.
            cargo += cargo2
            cargos = np.vstack((cargo, dest_cargo, dest.cargos[1:]))

        # Update route, times, block_time and profit
        route = [self.airpt] + dest.route
        times = [self.time] + dest.times
        block_time += dest.block_time
        profit += dest.profit


        return (profit, (route, times, cargos, demand, dest, block_time)), status

    def update_profit(self, cargo):
        cargo_diff = cargo - self.cargos[0]
        revenue_diff = yield_coeff*d*cargo_diff.sum()/1000
        self.cargos[0] = cargo
        self.profit += revenue_diff

    # Define update function to load new values into the node
    def update(self, profit, route, times, cargos, demand, dest, block_time): #total block time so far (part of system state)
        self.demand = demand
        self.route = route
        self.times = times
        self.cargos = cargos
        self.profit = profit
        self.next: Node = dest
        self.block_time = block_time

    def __str__(self):
        s = f"Node: (airpt: {self.airpt}, time: {self.time})\n"
        dest = self.next
        if dest is not None:
            ds = str(dest.airpt)
        else:
            ds = "None"
        s+= f"profit: {self.profit}\nnext: {ds}"
        return s

# %%
# Main loop
# Initialize results
ac_res = []

profit_res = []
route_res = []
times_res = []
cargos_res = []
demand_res = total_demand.copy()
block_res = []

stop = False
while not stop:
    # Initialize optimal results
    opt_profit = -np.inf
    opt_route = []
    opt_times = []
    opt_cargos = []
    opt_demand = []
    opt_block = 0
    opt_ac_type = None

    #fig, ax = plt.subplots(3,1)
    for ac_type in range(len(fleet)):
        # Check if there are aircraft of this type left available lease
        if fleet[ac_type] > 0:
            ac = Aircraft(ac_type)
            # Create network of nodes
            network = [[Node(ap, t) for t in range(time_steps)] for ap in range(AP)]
            
            #configure end node
            end: Node = network[hub][-1]
            end.demand = demand_res.copy()
            end.profit = 0
            print("network initialized")
            for t in range(time_steps-1):
                for ap in range(AP):
                    origin: Node = network[ap][time_steps-t - 2]
                    # Calculate results for each destination airport
                    res = [origin.calc_profit(network, ac, dest_airpt) for dest_airpt in range(AP)]

                    # Choose destination with the maximum profit
                    opt = max(res, key = lambda x: x[0])

                    # if flight is not infeasible, update the origin node with optimal flight
                    if opt[0][0] > -np.inf:
                        (profit, (route, times, cargos, demand, dest, block_time)), status = opt
                        origin.update(profit, route, times, cargos, demand, dest, block_time)

            node_profits = [[network[ap][t].profit for t in range(time_steps)] for ap in range(AP)]

            fig, ax = plt.subplots(3,1)
            ax[0].imshow(node_profits, aspect = "auto", interpolation = "none")
            
            

            #start node is at 24h mark (0*24). By definition
            #of the Node class. It should give the best valid route
            #(valid block time)
            start: Node = network[hub][0]
            
            ax[0].plot(start.times, start.route, color = "red")
            ax[1].imshow(np.log(demand_df.to_numpy()), aspect = "auto", interpolation = "none")
            ax[2].imshow(start.cargos, aspect = "auto", interpolation = "none")

            # Update optimal results if the profit at the start node is higher than the current optimal profit
            if start.profit > opt_profit:
                opt_profit = start.profit
                opt_route = start.route
                opt_times = start.times
                opt_cargos = start.cargos
                opt_demand = start.demand.copy()
                opt_ac_type = ac_type
                opt_block = start.block_time
            plt.title(f"Aircraft route and max profit per node")

            
            plt.show()
            raise Exception("stoppe maar")

   # End program if profit is nexative, there are no aircraft left or the block time of the optimal route is less than the minimum block time
    if opt_profit < 0 or np.all(fleet == 0) or opt_block < min_block:
        stop = True
    else:
        #update results
        fleet[opt_ac_type] -=1

        demand_res = opt_demand

        ac_res.append(opt_ac_type)
        profit_res.append(opt_profit)
        route_res.append(opt_route)
        times_res.append(opt_times)
        cargos_res.append(opt_cargos)

print(ac_res, profit_res, route_res, times_res, cargos_res, sep = "\n")
# %%
