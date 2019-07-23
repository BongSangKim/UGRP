import numpy as np; #NumPy package for arrays, random number generation, etc
import matplotlib.pyplot as plt #for plotting
import pandas as pd

#Simulation window parameters
xMin=0;xMax=1000;
yMin=0;yMax=1000;
xDelta=xMax-xMin;yDelta=yMax-yMin; #rectangle dimensions
areaTotal=xDelta*yDelta;

#Point process parameters
lambda0=4*10**-6 #intensity (ie mean density) of the Poisson process

#Simulate a Poisson point process
numbPoints = np.random.poisson(lambda0*areaTotal);#Poisson number of points
xx = xDelta*np.random.uniform(0,1,numbPoints)+xMin;#x coordinates of Poisson points
yy = yDelta*np.random.uniform(0,1,numbPoints)+yMin;#y coordinates of Poisson points

df = pd.DataFrame(data = (xx,yy))
df.to_csv("./BSposition.csv", index = False, header = False)

