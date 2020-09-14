import gpflow
from gpflow.kernels import *
import BNQD
import numpy as np
import matplotlib.pyplot as plt
import csv
from bnqdflow import models, base, effect_size_measures, util, analyses
import scipy.stats as stats
import os
import warnings
import importlib
import toolbox as t
import PlottingToolbox as pt
from SpectralMixture import SpectralMixture
import seaborn as sns
from tensorflow_probability import distributions as tfd
f64 = gpflow.utilities.to_default_float
warnings.simplefilter('ignore')
importlib.reload(BNQD)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Hide tensorflow warnings.
np.random.seed(1)

# Create synthetic data
N           = 100
sigma       = .2
b           = 0.
d           = 3.
x           = np.linspace(-1.25, 1.25, N)
# remove center data for testing
remove_center_data = False
if remove_center_data:
      empty       = x[40:60]
      N           = N - len(empty)
      x           = np.delete(x, np.arange(40,60))
# end testing

f           = t.triple_trigonometry(x, d)
f           = t.diverging_discontinuity_mean_function(x, d)
y           = np.random.normal(loc=f, scale=sigma, size=N).reshape(N,1)


# Generate test points for prediction
xlim        = [-1.5,1.5]
xx          = np.linspace(xlim[0], xlim[1], 1000).reshape(1000, 1)  # (N, D)
mu          = t.shifting_discontinuity_mean_function(xx, d)
mu          = t.diverging_discontinuity_mean_function(xx, d)
mu          = t.triple_trigonometry(xx, d)

true_freqs  = [ [12/(2*np.pi), 25/(2*np.pi)], [(12+d)/(2*np.pi), (25+d)/(2*np.pi)] ]
#pt.plot_data(x, y, xx, mu, b=b)

# Data used by the control model and intervention model
x1, x2      = x[x <= b], x[x>b]
y1, y2      = y[x <= b], y[x>b]
xx1, xx2    = xx[xx <= b], xx[xx > b]
data        = [(x1, y1), (x2, y2)]
max_freq    = 5.
max_length  = 75.

# Kernels
Q               = 3 #t.find_optimal_Q(x, y, min_Q=1, max_Q=7, max_length=max_length, max_freq=max_freq, plot_BIC_scores=True)
sm              = SpectralMixture(Q=Q, max_length=max_length, max_freq=max_freq)
periodic        = Periodic(SquaredExponential())
damped_periodic = SquaredExponential(lengthscales=10.)*Periodic(SquaredExponential())
colours_prior   = ['#1a1835', '#15464e', '#2b6f39', '#757b33', '#c17a70', '#d490c6', '#c3c1f2', '#cfebef']
#pt.plot_kernel_spectrum(Q, sm, max_x=max_freq*1.2, title="Initial GMM spectral density", colours=colours_prior)
#plt.show()

# Define training parameters
optim             = gpflow.optimizers.Scipy()
max_iter          = 1000
# Plotting parameters
padding           = 0.
ylim              = (-1.7,2.)#(-4.7,5.2)#(-2.5,2.3)

# Iterate over discontinuity sizes
discontinuity_sizes   = np.arange(0, 10, step=1)
discontinuity_sizes   = np.arange(3, 4 , step=1)
discontinuity_funcs   = [#('diverging',t.diverging_discontinuity_mean_function)]
                         #,('shifting',t.shifting_discontinuity_mean_function)]
                         #,('weight',t.weight_discontinuity_mean_function)]
                          ('triple trigonometry', t.triple_trigonometry)]
bayes_factors, effect_sizes = [np.zeros((len(discontinuity_funcs), discontinuity_sizes.shape[0])) for i in range(2)]
effect_sizes_with_weights = np.zeros((len(discontinuity_funcs), discontinuity_sizes.shape[0]))
effect_sizes_KL = np.zeros((len(discontinuity_funcs), discontinuity_sizes.shape[0]))

for i, d in enumerate(discontinuity_sizes):
      for n, (name, func) in enumerate(discontinuity_funcs):
            # Obtain data with discontinuity
            f = func(x,d)
            y = np.random.normal(loc=f, scale=sigma, size=N).reshape(N,1)
            y1, y2            = y[x <= b], y[x > b]

            true_freqs_shifting = [[12 / (2 * np.pi), 25 / (2 * np.pi)], [(12 + d) / (2 * np.pi), (25 + d) / (2 * np.pi)]]
            true_freqs_diverging= [[12 / (2 * np.pi), 25 / (2 * np.pi)], [(12 - d) / (2 * np.pi), (25 + d) / (2 * np.pi)]]
            if name == 'shifting':
                  true_freqs = true_freqs_shifting
            else:
                  true_freqs = true_freqs_diverging
            true_freqs = None

            # Create spectral mixture kernel
            Q                 = 5 #t.find_optimal_Q(x.reshape(N,1), y, min_Q=1, max_Q=7, max_length=max_length, max_freq=max_freq, plot_BIC_scores=True)
            sm                = SpectralMixture(Q=Q, max_length=max_length, max_freq=max_freq)
            sm.kernels[0].frequency.prior = tfd.Gamma(f64(2.0), f64(1.0))
            # Run analysis Spectral Mixture
            a = analyses.SimpleAnalysis([(x1, y1), (x2, y2)], sm, b, share_params=False)
            a.train(verbose=True)

            log_bayes_factor = a.log_bayes_factor(verbose=True)
            #print('Discontinuity: ', d, '. Log bayes factor of: f(x) = sin((12+d)x)+0.66 cos((25+d)x)', log_bayes_factor.numpy())
            bayes_factors[n,i] = log_bayes_factor.numpy()

            #pt.plot_posterior_model_spectrum(a, Q, padding=padding, max_x = max_freq*1.2, true_freqs=true_freqs,ylim=ylim)
            #plt.show()
            pt.plot_synthetic_control_posterior_spectrum(a, Q, padding=padding, max_x = max_freq*1.2, true_freqs=true_freqs,ylim=ylim)#[-2.,1.5])
            plt.show()
            # print('Control')
            # for kernel in a.discontinuous_model.models[0].kernel.kernels:
            #       print('Index: ', kernel.index, '. Frequency: ',
            #             round(kernel.frequency.numpy(), 3))
            # print('intervention')
            # for kernel in a.discontinuous_model.models[1].kernel.kernels:
            #       print('Index: ', kernel.index, '. Frequency: ',
            #             round(kernel.frequency.numpy(), 3))
            e = t.EffectSizeGMM(a, mode='difference in means')
            effect_sizes[n,i] = e
            e_with_weight = t.EffectSizeGMM(a, mode='means and weights')
            effect_sizes_with_weights[n,i] = e_with_weight
            e_KL = t.EffectSizeGMM(a, mode='KullbackLeibler')
            effect_sizes_KL[n,i] = e_KL
            print('Discontinuity', d,'Effect size', e)

plot_bayes_factors = False
if plot_bayes_factors:
      sns.set(style="ticks")
      print(bayes_factors)
      plt.figure()
      for i, (name, function) in enumerate(discontinuity_funcs):
            plt.plot(discontinuity_sizes, bayes_factors[i], label=name)
      plt.xlabel("Discontinuity size d")
      plt.title("Bayes' factors of f(x)= sin(12x) + cos(25x) ")
      plt.legend()
      plt.show()

plot_effect_sizes = False
if plot_effect_sizes:
      # Difference in means
      sns.set(style="ticks")
      plt.figure()
      for i, (name, function) in enumerate(discontinuity_funcs):
            plt.plot(discontinuity_sizes, effect_sizes[i], label=name)
      plt.xlabel("Discontinuity size d")
      plt.title("Effect sizes with difference in means")
      plt.legend()
      plt.show()

      # Means and weights
      sns.set(style="ticks")
      plt.figure()
      for i, (name, function) in enumerate(discontinuity_funcs):
            plt.plot(discontinuity_sizes, effect_sizes_with_weights[i], label=name)
      plt.xlabel("Discontinuity size d")
      plt.title("Effect sizes with difference in means plus difference in weights")
      plt.legend()
      plt.show()

      # Symmetric Kullback-Leibler
      sns.set(style="ticks")
      plt.figure()
      for i, (name, function) in enumerate(discontinuity_funcs):
            plt.plot(discontinuity_sizes, effect_sizes_KL[i], label=name)
      plt.xlabel("Discontinuity size d")
      plt.title("Effect sizes of sum of Kullback Leibler divergence")
      plt.legend()
      plt.show()


# Run Cannonical Periodic analysis
#-------------------------------
# a2 = analyses.SimpleAnalysis([(x1, y1), (x2, y2)], periodic, b, share_params=False)
# a2.train()
# bayes_factor = a2.log_bayes_factor(verbose=True)
# cm = a2.continuous_model.model
# dm = a2.discontinuous_model
# dcm = a2.discontinuous_model.control_model
# dim = a2.discontinuous_model.intervention_model
# print("log marginal likelihoods:\n\tcontinuous model: {}\n\tdiscontinuous control model: {}\n"
#       "\tdiscontinuous intervention model: {}"
#       .format(cm.maximum_log_likelihood_objective(), dcm.maximum_log_likelihood_objective(), dim.maximum_log_likelihood_objective()))
# fig2, ax2 = plt.subplots(2,1,figsize=(15,10))
# a2.continuous_model.plot_regression(ax=ax2[0], padding=padding)
# ax2[0].set_title("Cannonical Periodic Continuous")
# a2.discontinuous_model.plot_regression(ax=ax2[1], padding=padding)
# ax2[1].set_title("Cannonical Periodic Discontinuous")
# plt.show()

# Run Damped Periodic analysis
#-------------------------------
# a3 = analyses.SimpleAnalysis([(x1, y1), (x2, y2)], damped_periodic, b, share_params=False)
# a3.train()
# bayes_factor = a.log_bayes_factor(verbose=True)
# cm = a3.continuous_model.model
# dm = a3.discontinuous_model
# dcm = a3.discontinuous_model.control_model
# dim = a3.discontinuous_model.intervention_model
# print("log marginal likelihoods:\n\tcontinuous model: {}\n\tdiscontinuous control model: {}\n"
#       "\tdiscontinuous intervention model: {}"
#       .format(cm.maximum_log_likelihood_objective(), dcm.maximum_log_likelihood_objective(), dim.maximum_log_likelihood_objective()))
# fig3, ax3 = plt.subplots(2,1,figsize=(15,10))
# a3.continuous_model.plot_regression(ax=ax3[0], padding=padding, ylim=ylim)
# ax3[0].set_title("Damped Periodic Continuous")
# a3.discontinuous_model.plot_regression(ax=ax3[1], padding=padding, ylim=ylim)
# ax3[1].set_title("Damped Periodic Discontinuous")
# plt.show()