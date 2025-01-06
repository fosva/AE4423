#%%
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

# Define paramters
#cap_i is the capacity for flight i
cap = flights.Capacity.to_numpy()
#add ficticious capacity. it is not used for the constraints, but will prevent indexerrors
np.append(cap, 0)

#fare_p is the average ticket price for itinerary p
fare = itineraries["Price [EUR]"].to_numpy()
#add ficticious itinerary
np.append(fare, 0)

#D_p is the demand for itinerary p
D = itineraries.Demand.to_numpy()

#b_pr is the recapture rate of passengers being relocated from it. p to it. r.
#one extra itinerary is added at the end. Can be accessed by -1. This is the ficticious itinerary.
b = np.zeros((len(itineraries)+1, len(itineraries)+1))
np.fill_diagonal(b, 1)
for row in recapture.to_numpy(dtype=float):
    it0 = round(row[0])
    it1 = round(row[1])
    b[it0, it1] = row[2]
b[:,-1]=1
print(b)

# Define sets
L = len(flights)
P = len(itineraries)

# define delta matrix (1 if flight (leg) i belongs to the path p; 0 otherwise)
delta = np.zeros((len(flights), len(itineraries)+1))
for i in num_flights:
    for p in num_itineraries:
        if itineraries.loc[p]["Flight 1"] == flights.index[i] or itineraries.loc[p]["Flight 2"] == flights.index[i]:
            delta[i,p] = 1

# define unconstrained demand: demand of flight i is the sum
# of the demand of all itineraries that contain flight i
Q = np.zeros(len(flights))
for i in num_flights:
    Q[i] = sum(delta[i,p] * D[p] for p in num_itineraries)
#%%
"""
recapture_rate = np.zeros((len(itineraries), len(itineraries)))
for i in num_itineraries:
    recapture_rate[i,0] = 1
    recapture_rate[i,i] = 1
"""
#define which columns are used
#column -1 is the ficticious itinerary
usedcols=[-1]



# Step 2: Initialize the Gurobi model
start_time = time.time()
model = Model("Passenger_Mix")
print("Model initialized")


# Step 3: Create decision variables
t = model.addMVar((P,P), vtype=GRB.INTEGER, lb=0, name="t")
model.update()
print("Decision variables created")
var_time = time.time()
print(f"Time to create decision variables: {var_time - start_time}")


# Step 4: Objective function
model.setObjective(sum([t[p,r]*(fare[p]-b[p,r]*fare[r]) for p in range(P) for r in usedcols]), GRB.MINIMIZE)
print("Objective function created")
objective_time = time.time()
print(f"Time to create objective function: {objective_time - var_time}")


# Step 5: Add constraints
# Capacity constraints for each flight leg
flight_constraint = model.addConstr(sum([delta[:,p]*(t[p,r]-b[r,p]*t[r,p])+delta[i,r]*(t[r,p]-b[p,r]*t[p,r]) for p in range(P) for r in usedcols if p not in usedcols]) >= Q - cap, "C1: Cover overflow")
print("Capacity constraints added")

# Demand constraints for each itinerary
model.addConstr(t.sum(axis=1) <= D, "demand constraint")
print("Demand constraints added")

constraint_time = time.time()
print(f"Time to add constraints: {constraint_time - objective_time}")

# Update the model
model.update()
model.write('2A.lp')

# Set time constraint for optimization (5minutes)
model.setParam('TimeLimit', 1 * 10)
model.setParam('MIPgap', 0.009)



# Step 6: Solve the model
model.optimize()
#%%

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

#%%
#Print results
tres = t.getAttr('x')


# %%
