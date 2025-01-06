import numpy as np
import pandas as pd
from gurobipy import Model, GRB, quicksum
import time

# Step 1: Load data from GroupX.xlsx
file = "Group_35.xlsx"
flights = pd.read_excel(file, sheet_name="Flights",index_col=0)  # Flight schedule
itineraries = pd.read_excel(file, sheet_name="Itineraries")  # Passenger itineraries
recapture = pd.read_excel(file, sheet_name="Recapture")  # Recapture rates
num_flights = range(len(flights))
num_itineraries = range(len(itineraries))
num_recaptures = range(len(recapture))

# define delta function (1 if flight (leg) i belongs to the path p; 0 otherwise)
# def delta(i: str, p):
#     if itineraries.loc[p]["Flight 1"] == i or itineraries.loc[p]["Flight 2"] == i:
#         return 1
#     else:
#         return 0
delta = np.zeros((len(flights), len(itineraries)))
for i in num_flights:
    for p in num_itineraries:
        if itineraries.loc[p]["Flight 1"] == flights.index[i] or itineraries.loc[p]["Flight 2"] == flights.index[i]:
            delta[i,p] = 1

# define unconstrained demand: demand of flight i is the sum of the demand of all itineraries that contain flight i
# def unconstrained_demand(i):
#     Q = quicksum(delta(i,p) * itineraries.loc[p]["Demand"] for p in num_itineraries)
#     return Q
unconstrained_demand = np.zeros(len(flights))
for i in num_flights:
    unconstrained_demand[i] = sum(delta[i,p] * itineraries.loc[p]["Demand"] for p in num_itineraries)


# Define find recapture rate find the recapture rate for passengers rerouted from itinerary p to itinerary r
# def find_recapture_rate(p, r):
#     if not RMP:
#         # Use boolean indexing to find the row that matches the criteria
#         match = recapture[(recapture["From Itinerary"] == p) & (recapture["To Itinerary"] == r)]
#
#         # If a match is found, return the recapture rate, otherwise return 0
#         if not match.empty:
#             return match["Recapture Rate"].values[0]
#         return 0.0
#     if RMP:
#         if r == 0:
#             return 1
#         elif p == r:
#             return 1
#         else :
#             return 0
# recapture_rate = np.zeros((len(itineraries), len(itineraries)))
# for p in num_itineraries:
#     for r in num_itineraries:
#         recapture_rate[p,r] = find_recapture_rate(p,r)
recapture_rate = np.zeros((len(itineraries), len(itineraries)))
for i in num_itineraries:
    recapture_rate[i,0] = 1
    recapture_rate[i,i] = 1

start_time = time.time()
# Step 2: Initialize the Gurobi model
model = Model("Passenger_Mix")
print("Model initialized")
# Step 3: Create decision variables
t = {}
for p in num_itineraries:
    for r in num_itineraries:
        if r == 0 or p == r:
            t[p,r] = model.addVar(vtype=GRB.INTEGER, lb=0, name=f"t_{p}_{r}")
model.update()
print("Decision variables created")
var_time = time.time()
print(f"Time to create decision variables: {var_time - start_time}")
# Step 4: Objective function
model.setObjective(quicksum((itineraries.loc[p]["Price [EUR]"] - recapture_rate[p,0] * itineraries.loc[r]["Price [EUR]"])*t[p,r] for p in num_itineraries + quicksum(itineraries.loc[p]["Price [EUR]"] - recapture_rate[p,0] * 0*t[p,r] for p in num_itineraries)), GRB.MINIMIZE)
print("Objective function created")
objective_time = time.time()
print(f"Time to create objective function: {objective_time - var_time}")
# Step 5: Add constraints

# Capacity constraints for each flight leg
for i in num_flights:
#    model.addConstr(sum(x[i, j] for i in itineraries.index) <= flights.loc[flights['Flight No.'] == j, 'capacity'].values[0])
    model.addConstr(quicksum(delta[i,p]*t[p,r] for r in num_itineraries for p in num_itineraries) - quicksum(delta[i,p]*recapture_rate[r,p]*t[r,p] for p in num_itineraries for r in num_itineraries) >= unconstrained_demand[i]-flights.loc[flights.index[i]]["Capacity"], name=f"capacity_flight: {i}") #C1
print("Capacity constraints added")
# Demand constraints for each itinerary
for p in num_itineraries:
    model.addConstr(quicksum(t[p, r] for r in num_itineraries) <= itineraries.loc[p, 'Demand'], name=f"demand_itinerary: {p}") #C2
print("Demand constraints added")
# Recapture constraints
#for i in itineraries.index:
#    model.addConstr(s[i] == sum(r[i, k] for k in itineraries.index))
# Non-negativity constraints
for p in num_itineraries:
    for r in num_itineraries:
        model.addConstr(t[p, r] >= 0, name=f"non_negativity: t_{p}_{r}") #C3
print("Non-negativity constraints added")
constraint_time = time.time()
print(f"Time to add constraints: {constraint_time - objective_time}")
# Update the model
model.update()
model.write('2A.lp')
# Set time constraint for optimization (5minutes)
model.setParam('TimeLimit', 1 * 10)
model.setParam('MIPgap', 0.009)

# Recapture rate constraints
#for i in itineraries.index:
#    for j in itineraries.index:
#        if (i, j) in recapture.index:
#            recapture_rate = recapture.loc[(i, j), 'recapture_rate']
#            model.addConstr(r[i, j] <= recapture_rate * s[i])

# Step 6: Solve the model
model.optimize()


status = model.status

if status == GRB.Status.UNBOUNDED:
    print('The model cannot be solved because it is unbounded')

elif status == GRB.Status.OPTIMAL or True:
    f_objective = model.objVal
    print('***** RESULTS ******')
    print('\nObjective Function Value: \t %g' % f_objective)

elif status != GRB.Status.INF_OR_UNBD and status != GRB.Status.INFEASIBLE:
    print('Optimization was stopped with status %d' % status)
elif status == GRB.INFEASIBLE:
    model.computeIIS()
    model.write("model.ilp")
    print("Model is infeasible. Check model.ilp for details.")
else:
    print('Optimization was stopped with status %d' % status)

# Step 7: Output results
# Optimal objective value
if model.status == GRB.Status.OPTIMAL:
    print(f"Optimal Revenue: {model.objVal}")

    # Optimal decision variables for first 5 itineraries
    for p in num_itineraries[:5]:
        for r in num_itineraries:
            if t[p, r].X > 0:
                print(f"t[{p}, {r}] = {t[p, r].X}")

    # Optimal dual variables for first 5 flights
    for i in flights.index[:5]:
        constr = model.getConstrByName(f"capacity_flight: {i}")
        if constr:
            print(f"Flight {i}: Dual Value {constr.Pi}")
else:
    print("No optimal solution found.")

# Optimal decision variables for first 5 itineraries

# Optimal dual variables for first 5 flights

