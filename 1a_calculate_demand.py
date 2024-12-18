#%% 

import pandas as pd
import numpy as np
from math import sqrt, cos, sin, log, pi, e
from numpy import arcsin
import matplotlib.pyplot as plt

def debug(fn):
    def wrapper(*args, **kwargs):
        print(f"Invoking {fn.__name__}")
        print(f"  args: {args}")
        print(f"  kwargs: {kwargs}")
        result = fn(*args, **kwargs)
        print(f"  returned {result}")
        return result
    return wrapper

#%%
radian = pi/180
airports = pd.read_csv("airport_data.csv", sep='\t', index_col=0)


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

distances = pd.DataFrame(data = dist.round(2),
                         index = airports.axes[1],
                         columns = airports.axes[1])
distances.to_csv("distances.csv", sep="\t")

#%% Least squares estimation
pop = pd.read_csv("pop.csv", sep="\t", index_col=0)
gdp = pd.read_csv("gdp.csv", sep="\t", index_col=0)
demand = pd.read_csv("demand.csv", sep="\t", index_col=0)

years=2
f = 1.43 # fuel price
#matrix a will contain the values to do the least square estimation with.
#we do make the assumption that all cities/countries are indexed in the same order for all given data.
ac = [[1,
      log(pop.iloc[i,0]*pop.iloc[j,0]),
      log(gdp.iloc[i,0]*gdp.iloc[j,0]),
      -log(f*dist[i,j]),
      log(demand.iloc[i,j])] for i in range(n) for j in range(i+1,n)]
ac = np.array(ac)
a = ac[:,:-1]
c = ac[:,-1]

b = np.linalg.lstsq(a, c, rcond=None)
res = b[0]
res[0] = e**res[0]
print(res)
# %% Test results by plotting an image
#gravity model as function using estimated parameters.
def demand_forecast(popi, popj, gdpi, gdpj, f, d, est=res):
    #function from gravity model
    return est[0]*(popi*popj)**est[1]*(gdpi*gdpj)**est[2]/((f*d)**est[3])
xs = [[demand.iloc[i,j],
       demand_forecast(pop.iloc[i,0],
                       pop.iloc[j,0],
                       gdp.iloc[i,0],
                       gdp.iloc[j,0],
                       f, dist[i,j])]\
                       for i in range(n) for j in range(i+1,n)]
xs = np.array(xs)
plt.scatter(xs.T[0],xs.T[1])
M = demand.max(axis=None)
plt.plot([0,M],[0, M], color="orange")
plt.show()
# %%

#pop["2025"] = pop["2020"] + 5/3 * (pop["2023"]-pop["2020"])
#gdp["2025"] = gdp["2020"] + 5/3 * (gdp["2023"]-gdp["2020"])
# exponential growth instead of linear
pop_growth = (pop["2023"]/pop["2020"]) ** (1/3)
gdp_growth = (gdp["2023"]/gdp["2020"]) ** (1/3)

pop["2025"] = pop["2023"]*pop_growth**years
gdp["2025"] = gdp["2023"]*gdp_growth**years

# %%
pop_array = pop["2025"].to_numpy()
gdp_array = gdp["2025"].to_numpy()
demand_2025 = np.zeros((n,n))
for i in range(n):
    for j in range(n):
        if i != j:
            demand_2025[i][j] = demand_forecast(pop_array[i],
                                        pop_array[j],
                                        gdp_array[i],
                                        gdp_array[j],
                                        f, dist[i,j])
demand_2025
np.savetxt("demand_2025.csv", demand_2025, fmt = "%f", delimiter = ',')
# %%
