# Authors: Chi Zhang, Aaron Alphonsus
###############################################################################
import h5py
import numpy as np

from Forward import *
from MakeFigure import *

def WriteData(hdf5file, name, data):
    if name in hdf5file:
        del hdf5file[name]
    hdf5file.create_dataset(name, data=data)

filename = 'iceberg_data.h5'
hdf5file = h5py.File(filename, 'a')

t0 = 0.0
sig2 = 0.01

theta = [1, 0.1]
state0 = [308.0, 55.9, 0.5, 0.5]

tf = 43
T = 88

tobs = np.linspace(t0, tf, T)

[values, values2] = fixSecondMess()

# TODO: Don't copy x and y values into new arrays
ObsData = ForwardModel(tobs, theta, state0, values, values2)
xobs = [x[0] for x in ObsData]
yobs = [y[1] for y in ObsData]

# T = len(xobs)
# tobs = tobs[0:T]

cov = np.diag([sig2]*T)
x_noise = np.random.multivariate_normal([0.0]*T, cov)
y_noise = np.random.multivariate_normal([0.0]*T, cov)

x_data = xobs + x_noise
y_data = yobs + y_noise
WriteData(hdf5file, 'data/time', tobs)
WriteData(hdf5file, 'data/xobs', x_data)
WriteData(hdf5file, 'data/yobs', y_data)

time = np.linspace(t0, tf, 1000)
TrueData = ForwardModel(time, theta, state0, values, values2)
xtrue = [x[0] for x in TrueData]
ytrue = [y[1] for y in TrueData]

fig = MakeFigure(425, 0.9)
ax = plt.gca()
ax.plot(xtrue, ytrue, color = '#000cff')
ax.plot(x_data, y_data, 'o', markerfacecolor = '#00ffd8',
    markeredgecolor = '#00ffd8', markersize = 4)
ax.set_xlabel('Longitude', fontsize = 16, color = '#969696')
ax.set_ylabel('Latitude', fontsize = 16, color = '#969696')
plt.show()
