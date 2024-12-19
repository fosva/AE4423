import pandas as pd
from gurobipy import Model, GRB, quicksum

# Step 1: Load data from GroupX.xlsx
file = "Group_35.xlsx"
flights = pd.read_excel(file, sheet_name="Flights",index_col=0)  # Flight schedule
itineraries = pd.read_excel(file, sheet_name="Itineraries")  # Passenger itineraries
recapture = pd.read_excel(file, sheet_name="Recapture")  # Recapture rates
num_itineraries = range(len(itineraries))
num_recaptures = range(len(recapture))

# def find_recapture_rate(p,r):
#     for og in num_recaptures:
#         for alt in num_recaptures:
#             if recapture.loc[og]["From Itinerary"] == p and recapture.loc[alt]["To Itinerary"] == r:
#                 return recapture.loc[alt]["Recapture Rate"]
#     return 0.

# define delta function (1 if flight (leg) i belongs to the path p; 0 otherwise)
def delta(i: str, p):
    if itineraries.loc[p]["Flight 1"] == i or itineraries.loc[p]["Flight 2"] == i:
        return 1
    else:
        return 0
#print(delta("EA1007", 0))
# define unconstrained demand: demand of flight i is the sum of the demand of all itineraries that contain flight i
def unconstrained_demand(i):
    Q = quicksum(delta(i,p) * itineraries.loc[p]["Demand"] for p in num_itineraries)
    return Q
#print(unconstrained_demand("EA1007"))
#print(flights.loc["EA1007"]["Capacity"])
# Define find recapture rate find the recapture rate for passengers rerouted from itinerary p to itinerary r
def find_recapture_rate(p, r):
    # Use boolean indexing to find the row that matches the criteria
    match = recapture[(recapture["From Itinerary"] == p) & (recapture["To Itinerary"] == r)]

    # If a match is found, return the recapture rate, otherwise return 0
    if not match.empty:
        return match["Recapture Rate"].values[0]
    return 0.0

# Step 2: Initialize the Gurobi model
model = Model("Passenger_Mix")

# Step 3: Create decision variables
# x = model.addVars(itineraries.index, flights['Flight No.'], name="x", vtype=GRB.CONTINUOUS)
# s = model.addVars(itineraries.index, name="s", vtype=GRB.CONTINUOUS)
# r = model.addVars(itineraries.index, itineraries.index, name="r", vtype=GRB.CONTINUOUS)
t = {}
for p in num_itineraries:
    for r in num_itineraries:
        t[p,r] = model.addVar(vtype=GRB.INTEGER, lb=0, name=f"t_{p}_{r}")
#print (x,s,r)

# Step 4: Objective function
#revenue = sum(x[i, j] * itineraries.loc[i, 'fare'] for i in itineraries.index for j in flights['Flight No.'])
#spill_cost = sum(s[i] * itineraries.loc[i, 'fare'] * 0.5 for i in itineraries.index)  # Assume spill cost = 50% of fare
model.setObjective(quicksum((itineraries.loc[p]["Price [EUR]"] - find_recapture_rate(p,r) * itineraries.loc[r]["Price [EUR]"])*t[p,r] for p in num_itineraries for r in num_itineraries), GRB.MAXIMIZE)


# Step 5: Add constraints

# Capacity constraints for each flight leg
for i in flights.index:
#    model.addConstr(sum(x[i, j] for i in itineraries.index) <= flights.loc[flights['Flight No.'] == j, 'capacity'].values[0])
    model.addConstr(quicksum(delta(i,p)*t[p,r] for r in num_itineraries for p in num_itineraries) - quicksum(delta(i,p)*find_recapture_rate(r,p)*t[r,p] for p in num_itineraries for r in num_itineraries) >= unconstrained_demand(i)-flights.loc[i]["Capacity"], name=f"capacity_flight: {i}") #C1
# Demand constraints for each itinerary
print("Constraint 1 cleared")
for p in num_itineraries:
    model.addConstr(quicksum(t[p, r] for r in num_itineraries) <= itineraries.loc[p, 'Demand'], name=f"demand_itinerary: {p}") #C2

# Recapture constraints
for i in itineraries.index:
    model.addConstr(s[i] == sum(r[i, k] for k in itineraries.index))

# Recapture rate constraints
for i in itineraries.index:
    for j in itineraries.index:
        if (i, j) in recapture.index:
            recapture_rate = recapture.loc[(i, j), 'recapture_rate']
            model.addConstr(r[i, j] <= recapture_rate * s[i])

# Step 6: Solve the model
model.optimize()

# Step 7: Output results

# Optimal objective value
print(f"Optimal Revenue: {model.objVal}")

# Optimal decision variables for first 5 itineraries
for i in itineraries.index[:5]:
    print(f"Itinerary {i}: Spill {s[i].x}, Recapture {[r[i, j].x for j in itineraries.index]}, Flights {[x[i, j].x for j in flights['flight_number']]}")

# Optimal dual variables for first 5 flights
for j in flights['flight_number'][:5]:
    print(f"Flight {j}: Dual Value {flights.loc[flights['flight_number'] == j, 'capacity'].Pi}")
