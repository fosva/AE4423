import pandas as pd
import numpy as np
from math import sqrt, cos, sin, log
from numpy import arcsin

airports = pd.read_csv("airport_data.csv", sep='\t', index_col=0)
print(airports.loc["ICAO Code"])

def d(pos0: list, pos1: list):
    R = 6371
    #from the assignment:
    return 2*R*sqrt(sin((pos0[0]-pos1[0])/2)**2 + cos(pos0[0])*cos(pos1[0])*sin((pos0[1]-pos1[1])/2)**2)
print(airports.loc["Latitude (deg)", "Longitude (deg)"])
n = len(airports.axes[1])
D = np.zeros([n,n])
for i in range(n):
    for j in range(n):
        D[i,j] = d(airports.loc["Latitude (deg)", "Longitude (deg)"])


d = pd.DataFrame(index=airports.loc["ICAO Code"])

print(d.iloc[5,5])