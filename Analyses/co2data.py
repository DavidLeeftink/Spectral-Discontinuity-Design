import gpflow
f64 = gpflow.utilities.to_default_float
from gpflow.kernels import Linear, Matern12, Matern32, Matern52, Periodic, SquaredExponential, \
    Cosine, ArcCosine, RationalQuadratic, Exponential, Polynomial, White
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
import BNQD
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io
import os
import warnings
import importlib
import toolbox as t
from SpectralMixture import SpectralMixture


warnings.simplefilter('ignore')
importlib.reload(BNQD)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Hide tensorflow warnings.
#plt.style.use('seaborn-dark-palette')
np.random.seed(1)

# Plot data
def plot_data(xtrain, ytrain, xtest, ytest, b):
    plt.figure(figsize=(8, 5))
    plt.plot(xtrain, ytrain)
    plt.plot(xtest, ytest)
    plt.axvline(x=b, linestyle='--', linewidth=1, color='black', label='End of train data')
    plt.title('True function and noisy observations')
    plt.ylabel('Co2 concentration, ppm')
    plt.xlabel('Months after 1968 (should display years..)')
    #plt.xticks(np.arange(1968, 1968+(N/12), step=1/12))
    plt.legend()
    plt.show()

def plot_posterior_fit(X, Y, xx, mean, var, b, name):
    plt.figure(figsize=(8,5))
    plt.plot(X, Y, 'green',label='True data')
    plt.plot(xx, mean, 'C0',label='Posterior mean')

    # plt.fill_between(xx[:, 0],
    #                 mean[:, 0] - 1.96 * np.sqrt(var[:, 0]),
    #                 mean[:, 0] + 1.96 * np.sqrt(var[:, 0]),
    #                 color='C0', alpha=0.2)
    plt.axvline(x=N_train, color='black', linestyle=':')
    plt.title('Posterior model fit on C02 data for {} kernel'.format(name))
    plt.xlabel('years after 1968 (to do: fix the xticks..)')
    plt.ylabel('Co2 concentration, ppm')
    plt.legend()
    plt.show()

# Load in data
data = scipy.io.loadmat('datasets/CO2/CO2data.mat')
ytrain, ytest = data['ytrain'], data['ytest']
xtrain, xtest = data['xtrain'], data['xtest']


X_scalar = 12 # 1
xtrain, xtest = xtrain/X_scalar, xtest/X_scalar
xtrain, xtest = np.asfarray(xtrain), np.asfarray(xtest)
X = np.concatenate([xtrain, xtest])
Y = np.concatenate([ytrain, ytest])

N_train, N_test = ytrain.shape[0], ytest.shape[0]
N_train, N_test = N_train / X_scalar, N_test/X_scalar
N = int((N_train + N_test)*X_scalar)
xx = np.linspace(0, N/X_scalar, num=N).reshape(N, 1)
#plot_data(xtrain, ytrain, xtest, ytest, N_train)

# Standardize data: function taken from french municipality example
pre_standardization_mean = np.mean(ytrain)
pre_standardization_std  = np.std(ytrain)
zscorex = lambda x: (x - np.mean(x)) / np.std(x)
inverse_zscorex = lambda x: (x *pre_standardization_std) + pre_standardization_mean
ytrain = zscorex(ytrain)

# set up
max_iter = 1000
optim = gpflow.optimizers.Scipy()
n_burnin_steps = 100
n_samples = 1000
n_chains = 4

# Model 1: Compositional model from Rasmussen
# ------------
# 1. long term trend - Squared Exponential
# longterm_trend = SquaredExponential()
# longterm_trend.lengthscales.prior = tfd.Gamma(f64(30.0), f64(0.5))
# longterm_trend.variance.prior = tfd.Gamma(f64(30.0), f64(0.6))
# compositional_1 = longterm_trend
# # 2. seasonal trend - Cannonical periodic * Sqaured Exponential
# periodic_decay                              = SquaredExponential()
# periodic_decay.lengthscales.prior            = tfd.Gamma(f64(30.), f64(0.6))
# periodic_decay.variance.prior               = tfd.Gamma(f64(10.0), f64(0.8))
# periodic_trend                              = Periodic(SquaredExponential(lengthscales=10./X_scalar, variance=10./X_scalar), period=12./X_scalar)
# periodic_trend.base_kernel.lengthscales.prior       = tfd.Gamma(f64(10.), f64(0.1  * X_scalar))
# #periodic_trend.base_kernel.variance.prior          = tfd.Gamma(f64(10.), f64(0.1 * X_scalar))
# periodic_trend.period.prior                 = tfd.Gamma(f64(10.), f64(0.5 * X_scalar))
# compositional_1                             += periodic_trend #* periodic_decay



# 3. medium term irregularities -  Rational quadratic
# medium_irregularities                       = RationalQuadratic()
# #medium_irregularities.variance.prior        = tfd.Gamma(f64(2.0), f64(1.0 ))
# medium_irregularities.lengthscales.prior     = tfd.Gamma(f64(2.0), f64(1.0 ))
# medium_irregularities.alpha.prior           = tfd.Gamma(f64(2.0), f64(1.0 ))
# #compositional_1                             += medium_irregularities
# # 4. Noise - Squared Exponential + White noise
# noise                                       = SquaredExponential()
# noise.variance.prior                        = tfd.Gamma(f64(2.0), f64(1.0))
# noise.lengthscales.prior                    = tfd.Gamma(f64(2.0), f64(1.0))
# white_noise                                 = White()
# #white_noise.variance.prior                  = tfd.Gamma(f64(1.0), f64(1.0))
# #compositional_1                             += white_noise

# Run analysis: Periodic kernel multiplied with SE
# periodic_base = SquaredExponential()
# periodic_base.variance.prior = tfd.Gamma(f64(1.0), f64(1.0))
# periodic_base.lengthscales.prior = tfd.Gamma(f64(1.0), f64(1.0))
# periodic = Periodic(periodic_base)
# periodic.period.prior = tfd.Gamma(f64(2.0), f64(1.0))
# periodic.period.assign(2.0)
# gpflow.set_trainable(periodic.period, False)
# longterm_periodic = SquaredExponential(lengthscales=60.0)
# longterm_periodic.variance.prior = tfd.Gamma(f64(1.0), f64(1.0))
# longterm_periodic.lengthscales.prior = tfd.Gamma(f64(10.0), f64(.4))
# periodic = periodic * longterm_periodic
#
#
# model1 = t.train_exact_model(xtrain, ytrain, periodic, optim, max_iter)
# model1.likelihood.variance.prior = tfd.Gamma(f64(1.0), f64(1.0))
# gpflow.set_trainable(model1.likelihood.variance, False)
# #param_samples = t.HMC(model1, n_burnin_steps=n_burnin_steps, n_samples=n_samples, n_chains=n_chains)
# #density_kernels = t.kernel_density_estimation(param_samples)
# mean1, var1, samples1 = t.predict(model1, xx, n_samples = 20)
# mean1, var1, samples1 = inverse_zscorex(mean1), inverse_zscorex(var1), inverse_zscorex(samples1)
# plot_posterior_fit(X, Y, xx, mean1, var1, N_train, "Periodic")
# t.plot_posterior_sample(xx, samples1, name="Periodic")
# plt.show()
#
# # # # # # Run analysis:
# model2 = t.train_exact_model(xtrain, ytrain, compositional_1, optim, max_iter)
# #t.HMC(model2, n_burnin_steps=n_burnin_steps, n_samples=n_samples, n_chains=n_chains)
# mean2, var2, samples2 = t.predict(model2, xx, n_samples = 20)
# mean2, var2, samples2 = inverse_zscorex(mean2), inverse_zscorex(var2), inverse_zscorex(samples2)
# plot_posterior_fit(X, Y, xx, mean2, var2, N_train, "Compositional")
# t.plot_posterior_sample(xx, samples2, name="Compositional")
# plt.show()

#------------------
# Model 3: Spectral Mixture kernel with Q = 8, following analysis of Wilson-Adams (2013)
longterm_trend2 = SquaredExponential(lengthscales=60.)
longterm_trend2.lengthscales.prior = tfd.Gamma(f64(30.0), f64(0.35))
longterm_trend2.variance.prior = tfd.Gamma(f64(30.0), f64(0.6))
Q = t.find_optimal_Q(xtrain, ytrain, min_Q=1, max_Q=8, max_length=50., additional_kernel=longterm_trend2)
print('Q: ', Q)

spectral_mixture = SpectralMixture(Q=Q, max_length=650.) + longterm_trend2

model3 = t.train_exact_model(xtrain, ytrain, spectral_mixture, optim, max_iter)
model3.likelihood.variance.prior = tfd.Gamma(f64(1.0),f64(1.0))
#param_samples3 = t.HMC(model3, n_burnin_steps=n_burnin_steps, n_samples=n_samples, n_chains=n_chains)
#density_kernels3 = t.kernel_density_estimation(param_samples3)

mean3, var3, samples3 = t.predict(model3, xx, n_samples = 20)
mean3, var3, samples3 = inverse_zscorex(mean3), inverse_zscorex(var3), inverse_zscorex(samples3)
plot_posterior_fit(X, Y, xx, mean3, var3, N_train, f"Spectral Mixture (Q={Q})")
t.plot_posterior_sample(xx, samples3, name=f"Spectral Mixture (Q={Q})")
plt.show()
t.plot_kernel_spectrum(Q,spectral_mixture)




# SM from 2nd implementation
#------
# Q = 5 # number of mixtures
#
# # first get the sm kernel params set
# weights, means, scales = SpectralMixture.sm_init(train_x=X, train_y=Y, num_mixtures=Q)
# means += 0.01
# k_sm = SpectralMixture.SpectralMixture(num_mixtures=Q, mixture_weights=weights,mixture_scales=scales, mixture_means=means)
# k_sm_sum= k_sm + SquaredExponential()
# model4 = t.train_exact_model(xtrain, ytrain, k_sm_sum, optim, max_iter)
# #t.HMC(model3, n_burnin_steps=n_burnin_steps, n_samples=n_samples, n_chains=n_chains)
# mean4, var4, samples4 = t.predict(model4, xx, n_samples = 20)
# mean4, var4, samples4 = inverse_zscorex(mean4), inverse_zscorex(var4), inverse_zscorex(samples4)
# plot_posterior_fit(X, Y, xx, mean4, var4, N_train, "Spectral Mixture")
# t.plot_posterior_sample(xx, samples4, name="Spectral Mixture")
# plt.show()


