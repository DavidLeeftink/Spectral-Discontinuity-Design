import gpflow
f64 = gpflow.utilities.to_default_float
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
import BNQD
import matplotlib.pyplot as plt
import os
import warnings
import importlib
from gpflow.kernels import Linear, Matern12, Matern32, Matern52, Periodic, SquaredExponential, Cosine, ArcCosine, RationalQuadratic, Exponential, Polynomial
from SpectralMixture import SpectralMixture

from toolbox import train_exact_model, HMC, predict, mean_function, sinc_mean_function
from PlottingToolbox import plot_prediction, plot_covariance_matrix, plot_data, plot_prior_sample, plot_posterior_sample
import toolbox as t
import PlottingToolbox as pt

warnings.simplefilter('ignore')
importlib.reload(BNQD)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Hide tensorflow warnings.
#plt.style.use('seaborn-dark-palette')
#plt.style.use('ggplot')
np.random.seed(4)

print("GPFlow version:      {}".format(gpflow.__version__))
print("BNQD version:     {}".format(BNQD.__version__))
import seaborn as sns
#sns.set(style='white')
# Create synthetic data
# def mean_function(X):
#     return
N = 50
b = 0.5
n_samples_prior = 3
n_samples_posterior = 2
X = np.random.rand(N,1) #* 2 - 1
Y = mean_function(X) + np.random.randn(N,1)*0.1

# Generate test points for prediction
xlim = [-1,2]
xx = np.linspace(xlim[0], xlim[1], 1000).reshape(1000, 1)  # (N, D)
mu = mean_function(xx)
#plot_data(X, Y, xx, mu)
optim = gpflow.optimizers.Scipy()
max_iter = 1000

# Initialize kernels

linear          = Linear()
rbf             = SquaredExponential(lengthscales=0.5)
#rbf.lengthscales.prior = tfp.distributions.Gamma(f64(1), f64(2))
matern12        = Matern12()
matern12.lengthscales.prior = tfp.distributions.Gamma(f64(1), f64(2))
matern32        = Matern32()
polynomial      = Polynomial()
exp             = Exponential()
arccos          = ArcCosine()
periodic_se     = Periodic(SquaredExponential(), period = 0.5)
periodic_m52    = Periodic(Matern52())
gabor           = rbf * periodic_m52
Q = 2
spectral_mixture= SpectralMixture(Q=Q, max_freq=25.,max_length=10.)#,ytrain=Y)
#spectral_mixture = SquaredExponential()*Cosine() + SquaredExponential()*Cosine()

kernels = [rbf, periodic_se, spectral_mixture]
kernel_names = ['Squared Exponential', 'Periodic', 'Spectral Mixture']

# Train a model for each kernel, plot results.
import time
start = time.time()


fig, ax = plt.subplots(len(kernels), 2, sharex='col',figsize=(20,15))

models = []

for i, kernel in enumerate(kernels):
    #plot_covariance_matrix(kernel, ax[i,0],kernel_names[i])
    if i == 2:
        spectral_mixture2 = SpectralMixture(Q=Q, max_freq=10., max_length=2.)  # ,ytrain=Y)
        plot_prior_sample(xx, spectral_mixture2, 2, ax[i, 0], kernel_names[i], b=None)

    else:
        plot_prior_sample(xx, kernel, n_samples_prior, ax[i, 0], kernel_names[i], b=None)
    model = train_exact_model(X, Y, kernel, optim, max_iter)
    models.append(model)
    mean, var, samples = predict(model, xx, n_samples_posterior)
    plot_prediction(X, Y, xx, mean, var, samples, ax[i,1], kernel_names[i], b=None, mu = mu)

fig.savefig('prior_posterior_samples2.png')
plt.show()
#pt.plot_kernel_spectrum(Q, spectral_mixture)
#
# for i, kernel in enumerate(kernels):
#     m = models[i]
#     mean, var, samples = predict(m, xx, n_samples_posterior)
#     plot_posterior_sample(xx, samples,name=kernel_names[i], extra_width=1)
#     plt.show()
#     print(kernel.name, i)
