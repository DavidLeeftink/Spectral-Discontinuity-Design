
import GPy
import numpy as np
import matplotlib.pyplot as plt
import csv
import warnings; warnings.simplefilter('ignore')
import seaborn as sns
from rdd import rdd
import pandas as pd
import importlib
import sys
sys.path.append('../')
import BNQD
importlib.reload(BNQD)

print("GPy version:      {}".format(GPy.__version__))
print("BNQD version:     {}".format(BNQD.__version__))

plt.close('all')
sns.set()

print('Analysis of Heart rate data')

markers = ['o', 'd', '^', 's']
colors = ['#c70d3a', '#ed5107', '#230338', '#02383c']

windows = False
if windows:
    datafile = '..\\datasets\\Heart rate\\heartrate_dataset1.txt'
else:
    datafile = 'datasets/Heart rate/heartrate_dataset1.txt'

def read_data(datafile):
    print('Reading data')
    X = []
    file = open(datafile, "r")
    for i in file:
        k = float(i.split()[0])
        X.append(k)
    return X

def plot_timeseries(X):
    plt.plot(range(len(X)), X)
    plt.xlabel = "Time (msec)"
    plt.ylabel = "heart rate (?)"
    plt.show()

#X = read_data(datafile)
#plot_timeseries(X)

#--------------------------------
import mayavi
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

#Parameters to set
mu_x = 0
variance_x = 3

mu_y = 0
variance_y = 15

#Create grid and multivariate normal
x = np.linspace(-2,2,500)
y = np.linspace(-2,2,500)
X, Y = np.meshgrid(x,y)
pos = np.empty(X.shape + (2,))
pos[:, :, 0] = X; pos[:, :, 1] = Y
rv = multivariate_normal([mu_x, mu_y], [[variance_x, 0], [0, variance_y]])
z = np.cos(2*np.pi*((X - Y)))
#Make a 3D plot #['cividis']
cmaps =  [ 'Spectral', 'cividis', 'viridis']
            # 'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
            # 'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']
for cmapname in cmaps:
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, z,cmap=cmapname,linewidth=0.2)
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.set_ylim((np.min(z), np.max(z)+0.5))
    plt.show()


print('hello 1')



