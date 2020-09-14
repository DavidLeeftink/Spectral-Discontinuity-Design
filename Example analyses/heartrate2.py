import GPy
import numpy as np
import matplotlib.pyplot as plt
import csv
import math
import warnings; warnings.simplefilter('ignore')
import seaborn as sns
from rdd import rdd
import pandas as pd
import importlib
import sys
import bisect
sys.path.append('../')
import BNQD
import scipy.signal as signal
importlib.reload(BNQD)

# Load in data & plot
windows = False
if windows:
    datafile = '..\\datasets\\Heart rate\\heartrate_data2.csv'
else:
    datafile = 'datasets/Heart rate/heartrate_data2.csv'
dataset = pd.read_csv(datafile)

# first 8 heart beats only for faste runtime
n = len(dataset.hart.values)
n = 410
y = dataset.hart[:n].values

fs = 100
b = int(n/2)


# function taken from french municipality example
zscorex = lambda x: (x - np.mean(y)) / np.std(y)
x = zscorex(y)
x = np.linspace(-(n/5), n/5, n)
y = zscorex(dataset.hart[:n].values)
y[:int(n/2)-3] *= (0.8/1)

# Plot data
plt.title("Heart Rate Signal 2")
plt.plot(range(len(y)), y)
plt.xlabel("Time (10ms)")
plt.ylabel("Standardized measurement")
plt.show()

# Plot wavelet transform
range = [-int(n/2),int(n/2)]
widths = np.arange(1, 21)
cwtmatr = signal.cwt(y, signal.ricker, widths)
plt.imshow(cwtmatr, extent=[range[0], range[1], min(widths), max(widths)], cmap='PRGn', aspect='auto',
           vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())
plt.show()


# Plot periodogram
scale = "density"
freqs_sample, Pxx = signal.periodogram(x=y, fs=100, scaling=scale, axis=0)
plt.plot(freqs_sample, Pxx)
plt.show()

# Run BNQD
D = 1
x0 = 0

# Kernel choices
#linear_kernel = GPy.kern.Linear(D) + GPy.kern.Bias(D)
RBF_kernel = GPy.kern.RBF(D)
exp_kernel = GPy.kern.Exponential(D)
cos_kernel = GPy.kern.Cosine(D) + GPy.kern.Linear(D)
per_kernel = GPy.kern.PeriodicExponential(D)

kernel_names = [ 'Gaussian', 'Periodic exponential' ]
kernels = [ RBF_kernel, per_kernel]

kernel_dict = dict(zip(kernel_names, kernels))

# options to pass to BNQD
opts = dict()
opts['num_restarts'] = 50  # restarts for parameter optimization
opts['mode'] = 'BIC'  # method for approximating evidence
opts['verbose'] = True  # switch for more output

label_func = lambda x: x < x0  # point x < x0 are 'controls'

# do BNQD
qed = BNQD.BnpQedAnalysis(x, y, kernel_dict, label_func, b=x0, opts=opts)
qed.train()

# plot output
x_test = np.linspace(-(n/5), n/5, n)
qed.plot_model_fits(x_test=x_test)
plt.show()

bma_es = qed.get_total_BMA_effect_size()
print('BMA effect size: ',bma_es)
log_bayesfactor = qed.get_total_log_Bayes_factor()
print('Log Bayes factor: ', log_bayesfactor)