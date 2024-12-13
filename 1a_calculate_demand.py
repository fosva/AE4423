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
      log(pop.iloc[i,y]*pop.iloc[j,y]),
      log(gdp.iloc[i,y]*gdp.iloc[j,y]),
      -log(f*dist[i,j]),
      log(demand.iloc[i,j])] for i in range(n) for j in range(i+1,n) for y in range(years)]
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
       demand_forecast(pop.iloc[i,y],
                       pop.iloc[j,y],
                       gdp.iloc[i,y],
                       gdp.iloc[j,y],
                       f, dist[i,j])]\
                       for i in range(n) for j in range(i+1,n) for y in range(years)]
xs = np.array(xs)
plt.scatter(xs.T[0],xs.T[1])
M = demand.max(axis=None)
plt.plot([0,M],[0, M], color="orange")
plt.show()
# %%
