import gurobipy as gp
import pandas as pd

# Define sets: Airports, aircraft types
# Import airport data
Airports = pd.read_csv("airport_data.csv", sep='\t', index_col=0)
airports = len(Airports.iloc[2,:])

# Create aircraft dataframe
Aircraft_data = [["Aircraft type","Aircraft 1","Aircraft 2","Aircraft 3","Aircraft 4"],
                 ["Speed (km/h)", 550, 820, 850, 870],
                 ["Seats", 45, 70, 150, 320],
                 ["Average TAT", 25, 35, 45, 60],
                 ["Max Range (km)", 1500, 3300, 6300, 12000],
                 ["Runway required (m)", 1400, 1600, 1800, 2600],
                 ["Weekly lease cost (€)", 15000, 34000, 80000, 190000],
                 ["Fixed operating cost Cx (€)", 300, 600, 1250, 2000],
                 ["Time cost parameter CT (€/hr)", 750, 775, 1400, 2800],
                 ["Fuel cost parameter CF (€)", 1, 2, 3.75, 9],]
aircraft_data = pd.DataFrame(Aircraft_data[1:], columns = Aircraft_data[0])
aircraft_types = range(len(aircraft_data.iloc[0,:]))


# Define parameters TODO: Customize to assignment
CASK = 0.12 # Operation cost per ASK for aircraft type k TODO: incorporate lease cost etc (appendix B)
LF = 0.75 # Load Factor
s = 120 # number of seats per aircraft of type k (TODO: inorporate different aircraft types)
sp = 870 # speed of aircraft type k(TODO: incorporate different aircraft types)
LTO = 20/60 # Landing Time & Takeoff Time
BT = 10 * 7 # Block Time (average utilisation time)b of aircraft type k
AC = 2 # Number of aircrafts (TODO: Import from datashould be decision variable in our model)
Yield = 0.18  # yield TODO: make function of distance (appendix A)
hub = "EGLL"
f = 1.42 # EUR/gallon fuel cost
#for i in range(airports):
#    if Airports[1,i] == hub:
#        g_k = 0
#    else:
#       g_k = 1

# Define cost function based on appendix B
def leg_based_cost(aircraft_type: str, distance: float):
    C_x = aircraft_data.loc[aircraft_type, "Fixed operating cost Cx (€)"] # fixed operating cost for aircraft type used
    C_T = aircraft_data.loc[aircraft_type, "Time cost parameter CT (€/hr)"] * distance / aircraft_data.loc[aircraft_type, "Speed (km/h)"]  # Time-based costs for aircraft type and flight-leg used
    Fuel_cost = aircraft_data.loc[aircraft_type, "Fuel cost parameter CF (€/km)"] * f / 1.5 * distance  # Fuel costs for aircraft type and flight-leg used
    return C_x + C_T + Fuel_cost
print(leg_based_cost("Aircraft 1", 1000))

q = [[0, 1000, 200],  # Demand between airports (TODO: should be imported from data)
          [1000, 0, 300],
          [200, 300, 0]]
distance = [[0, 2236, 3201], # Distance between airports (TODO:should be imported from data)
          [2236, 0, 3500],
          [3201, 3500, 0]]

# Start modelling optimization problem
m = gp.Model('practice')

# Initialize decision variables
x = {} # direct flow between airports i and j
z = {} # number of flights between airports i and j for aircraft type k
w = {} #  flow rom airport i to airport j that transfers at hub
AC = {} # number of aircrafts of type k

# Define objective function
for i in airports:
    for j in airports:
        x[i, j] = m.addVar(obj=Yield * distance[i][j], lb=0, vtype=gp.GRB.INTEGER)
        w[i, j] = m.addVar(obj=Yield * distance[i][j], lb=0, vtype=gp.GRB.INTEGER)

        for k in aircraft_types:
            z[i, j, k] = m.addVar(obj=-CASK[k] * distance[i][j] * s[k], lb=0, vtype=gp.GRB.INTEGER)

m.update()
m.setObjective(m.getObjective(), gp.GRB.MAXIMIZE)  # The objective is to maximize revenue

# Define constraints
for i in airports:
    for j in airports:
        m.addConstr(x[i,j] + w[i][j] <= q[i][j]) #C1
        m.addConstr(w[i,j] <= q[i][j] * g[i] *g[j]) #C1*
        m.addConstr(x[i,j] <=z[i,j]*s*LF) #C2
    m.addConstr(gp.quicksum(z[i,j] for j in airports), gp.GRB.EQUAL, gp.quicksum(z[j, i] for j in airports)) #C3

m.addConstr(gp.quicksum(gp.quicksum((distance[i][j]/sp+LTO)*z[i,j] for i in airports) for j in airports) <= BT*AC) #C4


m.update()
# m.write('test.lp')
# Set time constraint for optimization (5minutes)
# m.setParam('TimeLimit', 1 * 60)
# m.setParam('MIPgap', 0.009)
m.optimize()
# m.write("testout.sol")
status = m.status

if status == gp.GRB.Status.UNBOUNDED:
    print('The model cannot be solved because it is unbounded')

elif status == gp.GRB.Status.OPTIMAL or True:
    f_objective = m.objVal
    print('***** RESULTS ******')
    print('\nObjective Function Value: \t %g' % f_objective)

elif status != GRB.Status.INF_OR_UNBD and status != GRB.Status.INFEASIBLE:
    print('Optimization was stopped with status %d' % status)


# Print out Solutions
print()
print("Frequencies:----------------------------------")
print()
for i in airports:
    for j in airports:
        if z[i,j].X >0:
            print(Airports[i], ' to ', Airports[j], z[i,j].X)