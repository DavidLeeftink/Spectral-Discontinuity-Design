# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 14:43:11 2020

@author: Max
"""

import GPy
import numpy as np
import matplotlib.pyplot as plt
import csv
import warnings;
warnings.simplefilter('ignore')
from rdd import rdd
import pandas as pd
import importlib
import sys
import scipy.signal as signal
import pywt
sys.path.append('../')
import BNQD

importlib.reload(BNQD)

print("GPy version:      {}".format(GPy.__version__))
print("BNQD version:     {}".format(BNQD.__version__))

plt.close('all')

print('Demo script for basic analysis with BNQD')

plot_data = True

# generate data
D = 1  # dimensionality
np.random.seed(1) # set seed for np gaussian samples
n = 50  # number of observations
range = [-10, 10]

x = np.linspace(range[0], range[1], n)  # predictor
x0 = 0  # point of intervention
b0 = 0 # bias
A = 0.5 # amplitude
f = 1 # frequency
phi = 0 # shift
sigma = 0.01  # noise level


#Synthetic functions for mean of underlying process
# 1. Simple cosine function for mean
#mu = b0 + A*np.sin(f*x + phi)  # true latent

# 2. Cosine with sudden change of frequency (used for shared hyperparam example)
mu = (np.sin((1/2)*x) + (1/2)*np.sin((3/2)*x)) * (x < x0) \
     + (np.sin((1/2)*x) + (1/2)*np.sin((3/2)*x))*(x > x0)
#mu = np.sin(x) + 0.66*np.cos(1.2*x) + 3
#mu = np.sin(x) / x
#mu = signal.waveforms.chirp(x, 0.02, 1, 0.04)
# chirp signal: mu = np.cos(2*np.pi * (3*x)) * np.exp(-np.pi*np.power(x,2)) + np.sin((2)*x)
# 3. Cosine with subperiodic behaviour
#mu = (np.sin(1*x) + np.cos(0.2*x)) * (x < x0) + (np.sin(1*x) + np.cos(0.2*x))*(x > x0)

#

# sample datapoints with Gaussian noise
y = np.random.normal(loc=mu, scale=sigma)  # observations

# Plot wavelet transform
# widths = np.arange(1, 31)
# cwtmatr = signal.cwt(y, signal.ricker, widths)
# plt.imshow(cwtmatr, extent=[range[0], range[1], min(widths), max(widths)], cmap='RdBu', aspect='auto',
#            vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())
# plt.show()
#
#
# w=16
# cwtm = signal.cwt(mu, signal.morlet2, widths, w=w)
# plt.imshow(cwtmatr, extent=[range[0], range[1], min(widths), max(widths)], cmap='RdBu', aspect='auto',
#            vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())
#plt.show()

# Plot periodogram
# scale = "density"
# freqs_sample, Pxx = signal.periodogram(x=y, fs=100, scaling=scale, axis=0)
# plt.plot(freqs_sample, Pxx)
# plt.show()

# for different true effect sizes: (note: currently not used)
for d_true in [0.2]:  # 0.2, 0.5, 1.5

    if plot_data:
        # show data
        plt.figure()
        plt.plot(x, mu, color='blue', linewidth=2, label='True function')
        plt.plot(x, y, linestyle='none', ms=6, marker='x', color='black', label='Observations')
        plt.axvline(x=x0, linestyle='--', linewidth=1, color='black', label='Intervention point')
        plt.title('True function and noisy observations')
        plt.legend()
        plt.show()

    # make a dictionary of kernels and their names
    linear_kernel = GPy.kern.Linear(D) + GPy.kern.Bias(D)
    RBF_kernel = GPy.kern.RBF(D)
    #per_kernel = GPy.kern.StdPeriodic(D, n_freq=2)
    sde_per_kernel = GPy.kern.sde_StdPeriodic(D)
    per_exp_kernel = GPy.kern.PeriodicExponential(D)
    per_mat_kernel = GPy.kern.PeriodicMatern32(D)
    per_lin_kernel = GPy.kern.StdPeriodic(D) + GPy.kern.Linear(D) + GPy.kern.Bias(D)

    #per_kernel = GPy.kern.periodic(D, n_freq = 4) + GPy.kern.Linear(D)
    # GPy.kern.ChangePointBasisFuncKernel
    cosine_kernel = GPy.kern.Cosine(D, lengthscale = 0.1) + GPy.kern.Cosine(D, lengthscale = 3)

    kernel_names = [ 'Gaussian', 'Cosine mixture']
    kernels = [ RBF_kernel, cosine_kernel]

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
    x_test = np.linspace(range[0], range[1], 100)  # for smoother plots

    qed.plot_model_fits(x_test=x_test)
    plt.show()
    #qed.plot_effect_sizes()
    #plt.show()
    #qed.plot_posterior_model_probabilities()
    #plt.show()
    print('done')

total_log_bayes_factor = qed.get_total_log_Bayes_factor(verbose=True)
total_bma_es = qed.get_total_BMA_effect_size()
print('total bma es :', total_bma_es)
print('total log bayes factor: ', total_log_bayes_factor)
