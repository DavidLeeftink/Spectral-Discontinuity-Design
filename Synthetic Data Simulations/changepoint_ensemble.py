import gpflow
import numpy as np
import BNQD
import matplotlib.pyplot as plt
import os
import warnings
import importlib

warnings.simplefilter('ignore')
importlib.reload(BNQD)
#import gp_samples
import scipy.signal as signal
from gpflow.kernels import SquaredExponential, Cosine, ChangePoints

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Hide tensorflow warnings.
np.random.seed(4)


def plot_data(X, Y, mu, ax):
    ax.plot(X, Y, color='red', label='Noisy samples')
    ax.plot(X, mu, color="black", label="True function")
    ax.set_title("True function and noisy sampled values")
    ax.legend()
    ax.set_ylim((-1.5, 1.5))


def plot_posterior(xx, mean, var, samples, ax, name = "non-stationary",locations=None):
    # plt.plot(X, Y, color='black', label='Noisy observation')
    ax.plot(xx, mean, 'C0', lw=2, label="Mean of posterior distribution")
    ax.fill_between(xx[:, 0],
                     mean[:, 0] - 1.96 * np.sqrt(var[:, 0]),
                     mean[:, 0] + 1.96 * np.sqrt(var[:, 0]),
                     color='C0', alpha=0.2)
    ax.plot(xx, samples[:, :, 0].numpy().T, 'C0', linewidth=.5, label="Samples drawn from posterior")
    if locations is not None:
        for c in locations:
            ax.axvline(x=c, color='black', linestyle=':')
    ax.set_title("Posterior prediction using the " + name + " kernel")
    ax.set_ylim((-1.5, 1.5))

def plot_fitted_mean(X, mean, true_mu, ax):
    ax.plot(X, true_mu, color="black", label="True function")
    ax.plot(X, mean, color="darkgreen", label="Predicted mean")
    ax.set_title("True function and estimated mean")
    ax.legend()
    ax.set_ylim((-1.5, 1.5))

def plot_hyperparam_over_time(lengthscales, variances, ax, thirdparam = None, title="Hyperparameters over time"):
    ax.plot(np.arange(len(lengthscales)), lengthscales, 'kx')
    ax.plot(np.arange(len(lengthscales)), lengthscales, label="Lengthscale")
    ax.plot(np.arange(len(variances)), variances, 'kx')
    ax.plot(np.arange(len(variances)), variances, label="Variance")
    if thirdparam is not None:
        ax.plot(np.arange(len(thirdparam)), thirdparam, 'kx')
        ax.plot(np.arange(len(thirdparam)), thirdparam, label="third parameter")
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.set_ylim((0,5))
    ax.legend()


# data specifications
N = 200
xlim = [0, 15]

# Generate data and test data
X = np.linspace(xlim[0], xlim[1], N)
mu = signal.waveforms.chirp(t=X, f0=0.01, t1=xlim[1], f1=1.)
#mu = 2.8*np.sin(X*2) / (X) # sinc
#mu = signal.morlet(N, w=10)
#mu = [float(x)*6 for x in mu]


#mu = np.power(np.pi,1/4) * np.power(np.exp(-0.5*(X^2)))
X = np.expand_dims(X, axis=1)
sigmas = [0.2, 0.4, 0.5]

for sigma in sigmas:
    fig, ax = plt.subplots(2,2, figsize=(12,8))
    ax = ax.ravel()
    Y = np.random.normal(loc=mu, scale=sigma).reshape(N,1)
    xx = np.linspace(xlim[0], xlim[1], N).reshape(N,1)
    plot_data(X,Y, mu, ax[0])

    # Model settings
    optim = gpflow.optimizers.Scipy()
    max_iter = 1000

    # Create c kernels, and assign c-1 changepoints to them
    K = 7
    # Assign linearly increasing lengthscales (specific to Chirp signal)
    #kernels = [Cosine(lengthscale = 4/(3*i+0.1)) for i in range(K)]

    kernels = [Cosine() for i in range(K)]
    #for kernel in kernels:
        #kernel.base.lengthscale.trainable = False
        #kernel.base.variance.trainable = False
    locations = [xlim[0]+(np.abs(xlim[1]-xlim[0])/K)*(i+1) for i in np.arange(K-1)]
    steepness = 1.
    non_stationary = ChangePoints(kernels, locations, steepness)
    non_stationary.locations.trainable = True
    #non_stationary.steepness.trainable = False
    non_stationary = custom_kernels.Gibbs(SquaredExponential())#*gpflow.kernels.ArcCosine()
    model = gp_samples.train_exact_model(X, Y, non_stationary, optim, 1000)
    mean, var, samples = gp_samples.predict(model, xx, n_samples=5)

    # Plot results
    #learned_locations = non_stationary.locations.numpy()
    plot_posterior(xx, mean, var, samples, ax[1])#, locations=learned_locations)
    plot_fitted_mean(xx, mean, mu, ax[2])


    # Plot learned hyperparameters over time (for sine)
    #learned_lengthscales = [kernel.kernels[1].lengthscale.numpy() for kernel in non_stationary.kernels]#kernel.kernels[0] is the cosine part
    #learned_variances = [kernel.kernels[1].variance.numpy() for kernel in non_stationary.kernels]
    #learned_period = [kernel.period.numpy() for kernel in non_stationary.kernels]
    #plot_hyperparam_over_time(learned_lengthscales, learned_variances, ax[3])
    plt.show()
    # Plot learned hyperparameters over time (for periodic)
    # learned_lengthscales = [kernel.base.lengthscale.numpy() for kernel in non_stationary.kernels]
    # learned_variances = [kernel.base.variance.numpy() for kernel in non_stationary.kernels]
    # learned_periods = [kernel.period.numpy() for kernel in non_stationary.kernels]
    # plot_hyperparam_over_time(learned_lengthscales, learned_variances, learned_periods)




