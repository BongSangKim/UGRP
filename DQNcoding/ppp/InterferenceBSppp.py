import numpy as np; #NumPy package for arrays, random number generation, etc
#import matplotlib.pyplot as plt #for plotting
import pandas as pd

#Simulation window parameters
xMin=1000;xMax=3000;
yMin=1000;yMax=3000;
xDelta=xMax-xMin;yDelta=yMax-yMin; #rectangle dimensions
areaTotal=xDelta*yDelta;

#Point process parameters
lambda0=32*10**-6 #intensity (ie mean density) of the Poisson process

#Simulate a Poisson point process
numbPoints = np.random.poisson(lambda0*areaTotal);#Poisson number of points
xx = xDelta*np.random.uniform(0,1,numbPoints)+xMin;#x coordinates of Poisson points
yy = yDelta*np.random.uniform(0,1,numbPoints)+yMin;#y coordinates of Poisson points

df = pd.DataFrame(data = (xx,yy))
df.to_csv("./InterferenceBSposition.csv", index = False, header = False)

