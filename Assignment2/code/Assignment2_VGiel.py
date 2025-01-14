#%%
import pandas as pd
import numpy as np
import os

# Read the excel files
Airports = pd.read_excel("AirportData.xlsx", index_col=0)
FleetOptions = pd.read_excel("FleetType.xlsx", index_col=0)
Demand_df = pd.read_excel("Group35.xlsx", skiprows=range(1,5), index_col=0)

#%%
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
demand = np.zeros((len(Airports.columns), len(Airports.columns),30))
for i in range(len(Airports.columns)):
    for j in range(len(Airports.columns)):
        for k in range(30):
            df_index = i * 20 + j + 1
            demand[i,j,k] = Demand_df.loc[df_index, str(k)]
# For example the demand from Airport 0 (LHR) to Airport 1 (CDG) for the 5th time slot
#print(Demand[0,1,5])
# parameters
hub = 3 # hub airport index (FRA)
max_steps = 1200 # maximum number of timesteps
num_airports = len(Airports.columns) # number of airports
num_fleet = len(FleetOptions.columns) # number of fleet types
f = 1.42 # fuel cost
Yield = 0.26 # yield in € per RTK

# Create array with distances between all airports
radian = np.pi/180
def d(pos0: list, pos1: list):
    R = 6371
    #from the assignment:
    return 2*R*np.sqrt(np.sin((pos0[0]-pos1[0])/2)**2 + np.cos(pos0[0])*np.cos(pos1[0])*np.sin((pos0[1]-pos1[1])/2)**2)

dist = np.zeros([num_airports,num_airports])
for i in range(num_airports):
    for j in range(num_airports):
        poss = Airports.loc[["Latitude (deg)", "Longitude (deg)"]].astype(float)
        loc0 = poss.iloc[:,i]
        loc1 = poss.iloc[:,j]
        dist[i,j] = d(loc0.to_numpy()*radian, loc1.to_numpy()*radian)

# Create array with costs for all legs
def leg_based_cost(aircraft_type: str, distance):
    C_x = FleetOptions.loc["Fixed Operating Cost (Per Fligth Leg)  [€]", aircraft_type] # fixed operating cost for aircraft type used
    C_T = FleetOptions.loc["Cost per Hour", aircraft_type] * distance / FleetOptions.loc["Speed [km/h]", aircraft_type]  # Time-based costs for aircraft type and flight-leg used
    Fuel_cost = FleetOptions.loc["Fuel Cost Parameter", aircraft_type] * f / 1.5 * distance  # Fuel costs for aircraft type and flight-leg used
    return C_x + C_T + Fuel_cost
costs = np.zeros([num_airports,num_airports,num_fleet])
for origin in range(num_airports):
    for destination in range(num_airports):
        for aircraft in range(num_fleet):
            costs[origin,destination,aircraft] = leg_based_cost(fleet_columns[aircraft],dist[origin,destination])

# define the node class
class Node:
    """
    The node class represents a node in the network of airports and timestep. The state of the node is determined by its airport and the timestep.
    Additionally, each node has a demand to other airports an optimal route for the rest of the week, a total profit for that optimal route, and a true/false whether it is the hub or not
    """
    def __init__(self, airport, timestep):
        """sd"""
        self.airport = airport # what airport the node is associated with
        self.timestep = timestep # what timestep the node is associated with
        self.demand = demand[airport,:,timestep//40] # the demand from this airport to all other airports at this timestep
        self.route = [] # the optimal route for the rest of the week updated at every timestep
        self.profit = 0 # the total profit of the optimal route
        self.is_hub = airport == hub # whether this node is the hub or not


#create the network of nodes
network = [[Node(airport, timestep) for timestep in range(max_steps)] for airport in range(num_airports)]

# define the inner loop, that given the origin node, the aircraft type, demand array calculates the to each airport, adds the profit of each destination airport and returns the maximum profit
def update_node(origin_node, aircraft_type, demand):
    origin = origin_node.airport
    timestep = origin_node.timestep
    timeslot = timestep//40
    opt_profit = 0
    if origin_node.is_hub:
        # loop over destinations
        for destination in range(num_airports):
            # calculate the distance
            distance = dist[origin, destination]
            travel_time = distance / FleetOptions.loc["Speed [km/h]", fleet_columns[aircraft_type]]
            block_time = travel_time + 0.5
            total_time = block_time + FleetOptions.loc["Average TAT [min]", fleet_columns[aircraft_type]] / 60
            arrival_timestep = timestep + np.ceil(total_time*10)


            # calculate cost
            cost = costs[origin, destination, aircraft_type]

            # determine the cargo flow
            max_potential_flow = demand[origin, destination, timeslot] + 0.2 * demand[origin, destination, timeslot - 1] + 0.2 * demand[origin, destination, timeslot - 2]
            Cargo_flow = min(max_potential_flow, FleetOptions.loc["Cargo capacity [kg]", fleet_columns[aircraft_type]])

            # calculate revenue (divide cargo flow by 1000 to convert kg to tonnes)
            revenue = Yield * Cargo_flow/1000 * distance # in €
            profit_leg = revenue - cost # profit for this leg from origin to destination
            if profit_leg > opt_profit:
                opt_profit  = profit_leg
                opt_destination = destination
                destination_node = network[destination][arrival_timestep]

        #update node with optimal profit and route
        node.profit = opt_profit
        opt_route = [opt_destination] + destination_node.route.copy()
        node.route = opt_route
        print("Node at airport", origin_node.airport, "at timestep", origin_node.timestep, "has optimal route", origin_node.route, "with profit", origin_node.profit)
    else: # if the node that is updated is not the hub, the destination is the hub by definition
        # calculate the distance
        distance = dist[origin, hub]
        travel_time = distance / FleetOptions.loc["Speed [km/h]", fleet_columns[aircraft_type]]
        block_time = travel_time + 0.5
        total_time = block_time + FleetOptions.loc["Average TAT [min]", fleet_columns[aircraft_type]] / 60
        arrival_timestep = timestep + np.ceil(total_time * 10)

        # calculate cost
        cost = costs[origin, hub, aircraft_type]

        # determine the cargo flow
        max_potential_flow = demand[origin, hub, timeslot] + 0.2 * demand[
            origin, hub, timeslot - 1] + 0.2 * demand[origin, hub, timeslot - 2]
        Cargo_flow = min(max_potential_flow, FleetOptions.loc["Cargo capacity [kg]", fleet_columns[aircraft_type]])

        # calculate revenue (divide cargo flow by 1000 to convert kg to tonnes)
        revenue = Yield * Cargo_flow / 1000 * distance  # in €
        profit_leg = revenue - cost  # profit for this leg from origin to destination





