#%% 

import pandas as pd
import numpy as np
from math import sqrt, cos, sin, log
from numpy import arcsin

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

airports = pd.read_csv("airport_data.csv", sep='\t', index_col=0)

@debug
def d(pos0: list, pos1: list):
    R = 6371
    #from the assignment:
    return 2*R*sqrt(sin((pos0[0]-pos1[0])/2)**2 + cos(pos0[0])*cos(pos1[0])*sin((pos0[1]-pos1[1])/2)**2)

n = len(airports.axes[1])
D = np.zeros([n,n])
for i in range(n):
    for j in range(n):
        poss = airports.loc[["Latitude (deg)", "Longitude (deg)"]].astype(float)
        loc0 = poss.iloc[:,i]
        loc1 = poss.iloc[:,j]
        D[i,j] = d(loc0.to_numpy(), loc1.to_numpy())

distances = pd.DataFrame(data = D.round(2),
                         index = airports.axes[1],
                         columns = airports.axes[1])
print(distances)
#%%
d = pd.DataFrame(index=airports.loc["ICAO Code"])

print(d.iloc[5,5])