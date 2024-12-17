import gurobipy as gp
import pandas as pd

# Define sets: Airports, aircraft types
# Import airport data
Airports = pd.read_csv("airport_data.csv", sep='\t', index_col=0)
airports = Airports.iloc[0]
num_airports = range(len(airports))


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
        x[i, j] = m.addVar(obj=revenue(distances.iloc[i,j]) * distances.iloc[i,j], lb=0, vtype=gp.GRB.INTEGER)
        w[i, j] = m.addVar(obj=revenue(distances.iloc[i,j]) * distances.iloc[i,j], lb=0, vtype=gp.GRB.INTEGER)

        for k in aircraft_types:
            z[i, j, k] = m.addVar(obj= -leg_based_cost(f"Aircraft {k+1}", distances.iloc[i,j]) , lb=0, vtype=gp.GRB.INTEGER)
for k in aircraft_types:
    AC[k] = m.addVar(obj= -aircraft_data.loc["Weekly lease cost (€)", f"Aircraft {k+1}"], lb=0, vtype=gp.GRB.INTEGER)

m.update()
m.setObjective(m.getObjective(), gp.GRB.MAXIMIZE)  # The objective is to maximize revenue

# Define constraints
for i in num_airports:
    for j in num_airports:
         m.addConstr(x[i,j] + w[i,j] <= demand.iloc[i,j], name="Demand verification") #C1
         m.addConstr(w[i,j] <= demand.iloc[i,j] * g(i) *g(j), name="Demand verification arriving at hub") #C1*
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
m.setParam('TimeLimit', 1 * 60)
m.setParam('MIPgap', 0.009)
m.optimize()
# m.write("testout.sol")
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