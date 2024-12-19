#%%
import gurobipy as gp
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import PIL
import numpy as np

# Define sets: Airports, aircraft types
# Import airport data
Airports = pd.read_csv("airport_data.csv", sep='\t', index_col=0)
airports = Airports.iloc[0]
num_airports = range(len(airports))
airport_coords = Airports.iloc[1:3,:].to_numpy().T
#print(airports[5])
#%%


# Create aircraft dataframe
aircraft_data = [["Speed (km/h)", 550, 820, 850, 870],
                 ["Seats", 45, 70, 150, 320],
                 ["Average TAT", 25/60, 35/60, 45/60, 60/60],
                 ["Max Range (km)", 1500, 3300, 6300, 12000],
                 ["Runway required (m)", 1400, 1600, 1800, 2600],
                 ["Weekly lease cost (€)", 15000, 34000, 80000, 190000],
                 ["Fixed operating cost Cx (€)", 300, 600, 1250, 2000],
                 ["Time cost parameter CT (€/hr)", 750, 775, 1400, 2800],
                 ["Fuel cost parameter CF (€)", 1, 2, 3.75, 9],]
aircraft_data = pd.DataFrame(aircraft_data, columns = ["Metric","Aircraft 1","Aircraft 2","Aircraft 3","Aircraft 4"])
aircraft_data.set_index("Metric", inplace=True) # Set index to metric
aircraft_types = range(len(aircraft_data.columns))

# Define parameters
LF = 0.75 # Load Factor
BT = 10 * 7 # Block Time (average utilisation time) of aircraft type k
hub = "EGLL"
f = 1.42 # EUR/gallon fuel cost

def g(index):
    airport = airports[index]
    if airport == hub:
        return 0
    else:
        return 1

# Define revenue function based on appendix A
def revenue(distance):
    if distance > 0.01:
        return 5.9 * (distance ** -0.76) + 0.043 # yield in RPK
    else:
        return 0.0

# Define cost function based on appendix B
def leg_based_cost(aircraft_type: str, distance: float):
    C_x = aircraft_data.loc["Fixed operating cost Cx (€)", aircraft_type] # fixed operating cost for aircraft type used
    C_T = aircraft_data.loc["Time cost parameter CT (€/hr)", aircraft_type] * distance / aircraft_data.loc["Speed (km/h)", aircraft_type]  # Time-based costs for aircraft type and flight-leg used
    Fuel_cost = aircraft_data.loc["Fuel cost parameter CF (€)", aircraft_type] * f / 1.5 * distance  # Fuel costs for aircraft type and flight-leg used
    return C_x + C_T + Fuel_cost

# Import demand and distance data
distances = pd.read_csv("distances.csv", sep='\t', index_col=0)
demand = pd.read_csv("demand_2025.csv", sep=',', header=None, names=airports)
demand.index = airports

# Big M for constraint C5 (range)
def a(i,j,k):
    if distances.iloc[i,j] <= aircraft_data.loc["Max Range (km)", f"Aircraft {k+1}"]:
        return 10000
    else:
        return 0

# Big M for constraint C6 (runway length)
def b(j,k):
    if float(Airports.iloc[3,j]) >= float(aircraft_data.loc["Runway required (m)", f"Aircraft {k+1}"]):
        return 10000
    else:
        return 0

# Function to increase TAT with 50% if destination is hub
def TAT(j,k):
    if airports[j] == hub:
        return 1.5 * aircraft_data.loc["Average TAT", f"Aircraft {k+1}"]
    else:
        return aircraft_data.loc["Average TAT", f"Aircraft {k+1}"]


# Start modelling optimization problem
m = gp.Model('practice')

# Initialize decision variables
x = {} # direct flow between airports i and j
z = {} # number of flights between airports i and j for aircraft type k
w = {} #  flow from airport i to airport j that transfers at hub
AC = {} # number of aircraft of type k

# Define objective function #

for i in num_airports:
    for j in num_airports:
        x[i, j] = m.addVar(obj=revenue(distances.iloc[i,j]) * distances.iloc[i,j]*1.0, lb=0, vtype=gp.GRB.INTEGER)
        w[i, j] = m.addVar(obj=revenue(distances.iloc[i,j]) * distances.iloc[i,j]*1.0, lb=0, vtype=gp.GRB.INTEGER)
        for k in aircraft_types:
            z[i, j, k] = m.addVar(obj= -leg_based_cost(f"Aircraft {k+1}", distances.iloc[i,j])/1. , lb=0, vtype=gp.GRB.INTEGER)
for k in aircraft_types:
    AC[k] = m.addVar(obj= -aircraft_data.loc["Weekly lease cost (€)", f"Aircraft {k+1}"], lb=0, vtype=gp.GRB.INTEGER)


m.update()
m.setObjective(m.getObjective(), gp.GRB.MAXIMIZE)  # The objective is to maximize revenue

# Define constraints
for i in num_airports:
    for j in num_airports:
        m.addConstr(x[i,j] + w[i,j] <= demand.iloc[i,j]*1.8, name="Demand verification") #C1
        m.addConstr(w[i,j] <= demand.iloc[i,j] * g(i) *g(j)*1.8, name="Demand verification arriving at hub") #C1*
        m.addConstr(x[i,j] + gp.quicksum(w[i,m] * (1-g(j)) for m in num_airports) + gp.quicksum(w[m,j] * (1-g(i)) for m in num_airports) <= gp.quicksum(z[i,j,k] * aircraft_data.loc["Seats", f"Aircraft {k+1}"] * LF for k in aircraft_types), name="Capacity verification") #C2
        for k in aircraft_types:
            m.addConstr(z[i,j,k] <= a(i,j,k), name="Range") #C5
            m.addConstr(z[i,j,k] <= b(j,k), name="Runway length") #C6

    for k in aircraft_types:
        m.addConstr(gp.quicksum(z[i,j,k] for j in num_airports) == gp.quicksum(z[j,i,k] for j in num_airports), name="Continuity/flow balance") #C3

for k in aircraft_types:
    m.addConstr(gp.quicksum(gp.quicksum((distances.iloc[i,j]/aircraft_data.loc["Speed (km/h)", f"Aircraft {k+1}"]+TAT(j,k))*z[i,j,k] for i in num_airports) for j in num_airports) <= BT*AC[k], name="AC productivity: Block time constraint") #C4


m.update()
m.write('model.lp')
# Set time constraint for optimization (5minutes)
m.setParam('TimeLimit', 1 * 10)
m.setParam('MIPgap', 0.009)
m.optimize()
m.write("testout.sol")
status = m.status

if status == gp.GRB.Status.UNBOUNDED:
    print('The model cannot be solved because it is unbounded')

elif status == gp.GRB.Status.OPTIMAL or True:
    f_objective = m.objVal
    print('***** RESULTS ******')
    print('\nObjective Function Value: \t %g' % f_objective)

elif status != gp.GRB.Status.INF_OR_UNBD and status != gp.GRB.Status.INFEASIBLE:
    print('Optimization was stopped with status %d' % status)
elif m.status == gp.GRB.INFEASIBLE:
    m.computeIIS()
    m.write("model.ilp")
    print("Model is infeasible. Check model.ilp for details.")
else:
    print('Optimization was stopped with status %d' % status)


# Print out Solutions
print()
print("Frequencies:----------------------------------")
print()

for i in num_airports:
    for j in num_airports:
        for k in aircraft_types:
            if z[i,j,k].X > 0:
                print(airports[i], ' to ', airports[j], 'with Aircraft', k+1, ":",z[i,j,k].X)
#i, j, k, l = 0, 0, 0, 0
# Print the values of all variables
# for i in m.getVars():
#     if i.X > 0:
#         print(i)
#         print(f"x: {i.VarName} = {i.X}")
for i in m.getVars():
    if i.varname == "C2400":
        print(f"Number of aircraft 1: {i.X}")
    elif i.varname == "C2401":
        print(f"Number of aircraft 2: {i.X}")
    elif i.varname == "C2402":
        print(f"Number of aircraft 3: {i.X}")
    elif i.varname == "C2403":
        print(f"Number of aircraft 4: {i.X}")
# print("Number of x variables: ", i)
# print("Number of z variables: ", j)
# print("Number of w variables: ", k)
# print("Number of AC variables: ", l)
#
# # Count and print the number of AC variables
# ac_vars = [var for var in m.getVars() if var.VarName.startswith('x')]
# print(f"Number of AC variables: {len(ac_vars)}")

#%%
G = nx.Graph()

# Add nodes to the graph
for i in num_airports:
    G.add_node(airports[i], pos=(float(airport_coords[i,1]), float(airport_coords[i,0]))) 

# Plot the locations as points

image = PIL.Image.open('europe.png')
x = [-25.11, 30.82]
y = [28.40, 66.55]
plt.figure(figsize=((x[1]-x[0])/5, (y[1]-y[0])/5))
plt.imshow(image, extent = [x[0], x[1], y[0], y[1]])
pos = nx.get_node_attributes(G, 'pos')
print(G, pos)
nx.draw(G, pos, with_labels=True, node_size=400, font_size=7, node_color='skyblue', font_weight='bold')


locations = airports
solution = {k: [] for k in aircraft_types}
#vehicle_demand = {k: 0 for k in aircraft_types} 
depot=0
# Collect the routes for each vehicle and track demand
for k in aircraft_types:
    for i in num_airports:
        for j in num_airports:
            if z[i, j, k].x > 0.5:  # Edge is part of the route
                solution[k].append((i, j,z[i,j,k].getAttr('x')))
                # Add the demand for the current location to the vehicle's total demand
                #if i != depot:
                #    vehicle_demand[k] += locations[i][3]  # Demand is stored at index 3 for each location


used_routes = np.zeros((len(airports), len(airports)))
# Plot the routes for each vehicle
for k in aircraft_types:
    vehicle_route = solution[k]
    has_legend = False
    for i, j,trips in vehicle_route:
        # Draw an edge from node i to node j for this vehicle's route.
        #using an offset to see routes used more than once.
        y1,x1 = np.array(airport_coords[i], dtype=float)
        y2,x2 = np.array(airport_coords[j], dtype=float)
        v = np.array([x2-x1,y2-y1])
        
        v_r = np.array([-v[1],v[0]])
        norm = np.linalg.norm(v)
        if norm == 0:
            print("Delen door 0 is flauwekul.")
            v_n = v_r
        else:
            v_n = v_r/norm

        for trip in range(round(trips)):
            offset = v_n*0.2
            visited_count = used_routes[i,j]
            p1 = np.array([x1,y1]) - offset*(0.8+visited_count)
            p2 = np.array([x2,y2]) - offset*(0.8+visited_count)
            used_routes[i,j] += 1
            plt.plot([p1[0],p2[0]], [p1[1],p2[1]], 
                        marker='o', linestyle='-', label=f"Aircraft type {k + 1}" if not has_legend else "", 
                        #color=plt.cm.get_cmap("tab10")(k))  # Assign a color to each vehicle
                        color=plt.colormaps.get_cmap("tab10")(k+1))
            has_legend = True

plt.legend()
plt.title('Fleet & Network model')
plt.grid(True)
plt.show()
# %%