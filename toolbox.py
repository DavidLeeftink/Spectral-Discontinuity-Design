# General
import os
import sys
import shutil
import posixpath
import warnings
import importlib
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Hide tensorflow warnings.
warnings.simplefilter('ignore')

# SciPy
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib.gridspec import GridSpec
import numpy as np
import scipy.stats as stats
from scipy.signal import periodogram, butter, filtfilt
from sklearn.neighbors import KernelDensity

# Gpflow & tensorflow
import tensorflow as tf
from tensorflow_probability import distributions as tfd
import gpflow
f64 = gpflow.utilities.to_default_float
from gpflow.kernels import *

# Files
import csv
import pandas as pd
import seaborn as sns
import wfdb
from scipy.io import loadmat 

# Local scripts
#sys.path.insert(0, '/home/david/Documenten/Courses/Spectral Discontinuity Design/Thesis AI/Implementation/')
from SpectralMixture import SpectralMixture, initialize_from_emp_spec, SpectralMixtureComponent
from bnqdflow import models, base, effect_size_measures, util, analyses


# """
# - Parts of the code inspired by: https://gpflow.readthedocs.io/en/stable/notebooks/regression.html
# - Plot of data & noisy observations taken from BNQD:
# """

def find_optimal_Q(x, y, data, min_Q, max_Q, fs=1., added_kernel = None, plot_BIC=True):
    """
    Determine the optimal number of spectral components according to the Bayesain Information Criterion
    BIC = log likelihood - k/2 * log(n)
    
    :param x:  NumpPy array (n,) input x values
    :param y:  NumPy array (n,) data y values
    :param data: GPRegression datatype.
    :param Qs: range of values for which a SM kernel will be intialized with Q components
    :param fs: sampling rate
    :plot_BIC: Bool: plots the BIC scores if true
    
    return number of components Q that 
    """
    Qs= np.arange(min_Q, max_Q)
    BIC = np.zeros((Qs.shape[0]))
            
    for i, q in enumerate(Qs):
        sm     = SpectralMixture(q, x=x.flatten(),y=y.flatten(),fs=fs)
        for k in sm.kernels:
            if isinstance(k, SpectralMixtureComponent):
                k.lengthscale.prior = tfd.Gamma(f64(8.), f64(.6))         
                k.mixture_weight.prior = tfd.Gamma(f64(2.), f64(1.))

        if added_kernel is not None:
            sm += added_kernel
                
#         model  = models.ContinuousModel(sm, (util.ensure_tf_matrix(x),util.ensure_tf_matrix(y)))
        model  = models.ContinuousModel(sm, data)
        model.train(verbose=False)
        BIC[i] = model.log_posterior_density("bic").numpy()
 
    if plot_BIC:
        fig = plt.figure()
        plt.plot(Qs, BIC)
        plt.xlabel('Number of Spectral Mixture components (Q)')
        plt.show()

    return np.argmax(BIC) + min_Q 

def BIC_scores(X, Y,min_Q=1, max_Q=10, max_freq=1.0, max_length=1.0):
    BIC_scores = np.zeros((max_Q-min_Q))

    for q in range(min_Q, max_Q):
        sm = SpectralMixture(Q=q, max_freq=max_freq, max_length=max_length)

        m = models.ContinuousModel(sm, (util.ensure_tf_matrix(X),util.ensure_tf_matrix(Y)))
        m.train(verbose=False)
        log_marginal_likelihood = m.log_posterior_density("bic").numpy()

        BIC_scores[q-min_Q] = log_marginal_likelihood
#        print(f'Q: {q}',np.round(log_marginal_likelihood,3), 'BIC: ',np.round(BIC_scores[q-min_Q] ,3) )

    return BIC_scores

def gaussian_density(x, weight, var, mean, log=True):
    """
    Gaussian probability density
    :param weight:
    :param var: variance, sigma^2
    :param mean:
    :return: 1/sqrt(2 pi var) * exp{-(x-mu)^2/2}
    """

    exp_term = np.exp(- (np.power ((x - mean)/var, 2.) /2))
    normalization_term = 1 / ( np.sqrt(2.*np.pi*var))
    log_gaussian = -np.log(np.sqrt(np.pi)) - np.log(var) - (x-mean)**2 / (2*var)
    return weight * exp_term * normalization_term, log_gaussian


#---------------------------
# Effect size measurements
#---------------------------

def EffectSizeGMM(analysis, mode='absolute difference'):
    """
    Effect size for the Spectral Mixture kernel
    :param analysis:        BNQDflow analysis object
    :param mode:            effect size measure. Currently supports 'difference in means' and 'kullback leibler'
    :return: effect size    \Sum_{i=0}^{N} (mu_{i,c} - mu_{i,d})
    """
    ControlSM               = analysis.discontinuous_model.models[0].kernel.kernels
    InterventionSM          = analysis.discontinuous_model.models[1].kernel.kernels
    Q                       = len(ControlSM)
    weights, means, sigmas  = [np.zeros((2,Q)) for i in range(3)]

    for i, kernel in enumerate([ControlSM, InterventionSM]):
        for j in range(Q):
            weights[i, j] = kernel[j].mixture_weight.numpy()
            means[i, j] = kernel[j].frequency.numpy()
            sigmas[i, j] = 1 / np.sqrt(kernel[j].lengthscale.numpy())

    if mode == 'absolute difference':
        return DifferenceInMeans(weights, means, sigmas)
    elif mode == 'KullbackLeibler':
        return EffectSizeKullbackLeibler(weights, means, sigmas)
    elif mode == 'means and weights':
        return DifferenceInMeans(weights, means, sigmas) + DifferenceInWeights(weights)
    else:
        raise ValueError("Effect size measure is misspecified.")


def DifferenceInMeans(weights, means, sigmas, average_delta = True):
    """
    Effect size as the difference in means, weight and standard deviations between two Gaussian Mixture Models.
    :arg weights    (2, Q) where the first index is the control group
    :arg means:     (2, Q) where the first index is the control group
    :arg sigmas:    (2, Q) where the first index is the control group
    :return float : Scaled difference in means, weight and standard deviations between the Control and Intervention spectral mixtures.
    """
    Q = len(means)
    delta = 0
    # Sort on frequency
    means.sort()

    # Multiply absolute difference in frequ

    for i in range(Q):
        delta += (np.abs((means[0,i])-(means[1,i]))
                + np.abs((weights[0,i])-(weights[1,i]))
                + np.abs((sigmas[0,i])-(sigmas[1,i])))
        #print('delta: ',delta, ' Q: ', Q)
    if average_delta:
        delta = delta/Q
    return delta

def EffectSizeKullbackLeibler(weights, means, sigmas):
    """
    Effect size according to the approximated Kullback-Leibler divergence for two univariate Gaussian Mixture Models.
    For each pair of Gaussians, the effect size as approximated as:
    D(m1||m2) = KL(m1||m2) = sum_Q argmin_j
    where KL(p|q) = log (sigma_1/sigma_2) + (sigma_1^2 +(mu_1 - mu_2)^2 )/ 2 sigma_2^2 - 1/2
    :arg weights,  array of (Q,)
    :arg means, array of (Q,)
    :arg sigmas, array of (Q,)

    :return float : approximated Kullback-Leibler divergence between Mixture1 and Mixture2.
    """
    kl = 0
    Q = len(weights)

    # Normalize weights
    sum_weights = np.sum(weights)
    weights = weights / sum_weights

    # Pair each component in the control group to the closest component in the intervention group
    for i in range(Q):
        m1, s1, w1 = means[0,i] , sigmas[0,i], weights[0,i]
        kl_scores = np.zeros((Q))
        for j in range(len(weights)):
            m2, s2, w2 = means[1,j], sigmas[1,j], weights[1,j]
            kl_univariate = KullbackLeiblerUnivariateGaussian(m1,m2,s1,s2)
            kl_scores[j] =  kl_univariate + np.log(w1/w2)

        kl += w1 * kl_scores.min()
    return kl

def KullbackLeiblerUnivariateGaussian(m1, m2, s1, s2):
    """
    :param m1, m2 : means
    :param s1, s2: standard deviation
    :return: KL divergence for univariate gaussian
        KL(p||q) = log (sigma_1/sigma_2) + (sigma_1^2 +(mu_1 - mu_2)^2 )/ 2 sigma_2^2 - 1/2
    """
    return np.log(s2/s1) +(s1**2 + (m1-m2)**2)/(2*s2**2) - 0.5


def DifferenceInWeights(weights, average_delta = True):
    Q = len(weights)
    delta = 0
    for i in range(Q):
        delta += np.abs((weights[0, i]) - (weights[1, i]))
        #print('delta: ', delta, ' Q: ', Q)
    if average_delta:
        delta = delta / Q
    return delta


#---------------------------
# Functions for generating synthetic data sets
#---------------------------

def mean_function(x):
    """
    f(x) = sin(12x) + (2/3)*cos(25x)
    """
    return np.sin(12*x) + 0.66*np.cos(25*x) # original frequencies: 12, 25

def shifting_discontinuity_mean_function(x, d=0):
    """
    f(x) = sin((12+d) x) + (2/3)cos((25+d) x) where forall x>0, d=0
    :arg d: order of discontinuity in frequency
    """
    return (x<=0)*np.sin(12*x) + (x>0)*np.sin((12+d)*x)+ (x<=0)*0.66*np.cos(25*x) + (x>0)*0.66*np.cos((25+d)*x)

def diverging_discontinuity_mean_function(x, d=0):
    """
    f(x) = sin((12-d)x) + (2/3)cos((25+d)x) where forall x>0, d=0
    :arg d: order of discontinuity in frequency
    """
    return (x<=0)*np.sin(12*x) + (x>0)*np.sin((12-d)*x)+ (x<=0)*0.66*np.cos(25*x) + (x>0)*0.66*np.cos((25+d)*x)

def weight_discontinuity_mean_function(x, d=0):
    """
    f(x) = sin(12x)/d + (2*d/3)cos(25x) where forall x>0, d=0
    :arg d: order of discontinuity in weight
    """
    return (x <= 0) * np.sin(12 * x) + (x > 0) *(1/d)* np.sin(12* x) + (x <= 0) * 0.66 * np.cos(25 * x) + (
                x > 0) * (0.66*d) * np.cos(25*x)

def sinc_mean_function(x):
    """
    f(x) = (sin(24x) / 12x) + 2
    """
    return np.sin(24*x ) / (12*x) + 2

def triple_trigonometry(x, d=0):
    """
    f(x) = sin(5x) + cos(17x) + sin(28x)
    :arg d: order of discontinuity
    """
    return (np.cos(5*x) +np.cos(17*x) + np.cos(28*x))*(x<=0) + (np.cos((5+d)*x) +np.cos((17+d)*x) + np.cos((28+d)*x))*(x>0)


#---------------------------
# Plotting functionality
#---------------------------

# def plot_prediction(X, Y, xx, mean, var, samples=None, ax=None, name=None, b=None, mu = None, ylim=None, lineplot=False):
#     """
#     Standard time series plot of the observations and the model fit.
#     :param X: observation X-values
#     :param Y: obsrvation Y-values
#     :param xx: NumPy linspace object for predictions
#     :param mean: posterior model mean
#     :param var: posterior model variance
#     """
#     if ax == None:
#         ax = plt.gca()
#     if name == None:
#         name = "undefined"

#     if lineplot:
#         ax.plot(X,Y,color='black')
#     else:
#         ax.plot(X, Y, 'kx',mew=2,label='Observations')
#     ax.plot(xx, mean, 'C0',linewidth=2.0,label='Posterior mean')
#     ax.fill_between(xx[:, 0],
#                     mean[:, 0] - 1.96 * np.sqrt(var[:, 0]),
#                     mean[:, 0] + 1.96 * np.sqrt(var[:, 0]),
#                     color='C0', alpha=0.15)#'C0'

#     ax.xaxis.set_major_locator(LinearLocator(4))
#     ax.yaxis.set_major_locator(LinearLocator(2))
#     ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
#     ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
#     ax.set_title("Posterior of " + name + " kernel", fontsize=25)
#     ax.set_ylim(-3,3)
#     if samples is not None:
#         ax.plot(xx, samples[:, :, 0].numpy().T, 'C0', linewidth=.5)
#     if mu is not None:
#         ax.plot(xx, mu, label='True function', linewidth=2.0, color='black')
#     if b is not None:
#         ax.axvline(x=b, color='black', linestyle=':')
#     ax.legend(loc='upper right')

#     #ax.set_title("Posterior prediction using the " + name + " kernel")
#     if ylim != None:
#         ax.set_ylim(ylim[0], ylim[1])

def plot_kernel_spectrum(spectral_mixture, max_x=1.1, title=None, ax=None, colours=None, true_freqs=None,
                         two_pi=False, scalar=1.):
    """
    Plot the estimated spectral density for the Spectral Mixture kernel
    :param spectral_mixture: spectra mixture kernel
    """
    Q = np.sum([isinstance(kernel, SpectralMixtureComponent) for kernel in spectral_mixture.kernels])
    
    if ax is None:
        ax = plt.gca()
    weights, means, variances = [], [], []
    for i in range(Q):
        weights.append(spectral_mixture.kernels[i].mixture_weight.numpy())
        if two_pi:
            means.append((spectral_mixture.kernels[i].frequency.numpy() / (2 * np.pi)) * scalar)
        else:
            means.append((spectral_mixture.kernels[i].frequency.numpy()) * scalar )
            
        variances.append(1 / np.sqrt(spectral_mixture.kernels[i].lengthscale.numpy() ))#* (1 / scalar)
#         print('Trained value lengthscale ', spectral_mixture.kernels[i].lengthscale.numpy())
#         print('Scaled value', np.sqrt(spectral_mixture.kernels[i].lengthscale.numpy() * (1/scalar)))
#         print('final variance: ', 1 / np.sqrt(spectral_mixture.kernels[i].lengthscale.numpy() * (1/scalar)))
    # weights = np.log(weights)
    # vars = np.log(vars)
    pdfs = plot_freq_GMM(weights, means, variances, title=title, max_x=max_x, ax=ax, colours=colours, true_freqs=true_freqs)
    return pdfs

def plot_freq_GMM(weights, means, variances, title=None, max_x=1.1, ax=None, colours=None, true_freqs=None):
    if ax is None:
        ax = plt.gca()
    x = np.linspace(0, max_x, int(max_x * 1000)).reshape(int(max_x * 1000), 1)

    max_val = 0
    pdfs = []
    for i in range(len(weights)):
        pdf, log_pdf = gaussian_density(x, weights[i], variances[i], means[i])
        pdfs.append(pdf)
        if max(pdf) > max_val:
            max_val = max(pdf)
        if colours is not None:

            ax.plot(x, pdf, linewidth=2.0, color=colours[i % 6], rasterized=True)
            ax.fill_between(x.flatten(), 0, pdf.flatten(), alpha=0.35, color=colours[i % 6], rasterized=True)
        else:
            ax.plot(x, pdf, linewidth=2.0, rasterized=True)
            ax.fill_between(x.flatten(), 0, pdf.flatten(), alpha=0.25, rasterized=True)
    if true_freqs is not None:
        for i, freq in enumerate(true_freqs):
            if i is 0:
                ax.axvline(x=freq, color='black', linestyle=':', label='True frequencies')
            else:
                ax.axvline(x=freq, color='black', linestyle=':')

    ax.set_xlabel("Frequency")  # , fontsize=16)
    ax.set_ylabel("Spectral Density")  # ,fontsize=16)
    if title is not None:
        ax.set_title(title, fontsize=25)
    ax.set_ylim(0, max_val + 0.5 * max_val)
    ax.set_xlim(0, max_x) # 30
    return np.array(pdfs)
    # ax.legend()


def plot_model_spectra(a, Q, d, ax, index, max_x=2.5, padding=0.0, true_freqs=None, ylim=None, scalar=1.0):
    """
    Plots several model spectra for various discontinuity sizes
    """
    # colour palettes
    greys = ['#393939', '#575757', '#707070', '#898989', '#a4a4a4', '#bfbfbf']
    blues = ['#367bac', '#3787c0', '#4892c6', '#69a6d0', '#8abbdb', '#95c1de']
    greens = ['#265e52', '#3c7e69', '#599d7e', '#79b895', '#9ed0ae', '#c6e5cc']

    sns.set(style='ticks')
    cm = a.continuous_model.model
    dcm = a.discontinuous_model.control_model
    dim = a.discontinuous_model.intervention_model

    #     fig = plt.figure(constrained_layout=True,figsize=(12,15))

    #     # Continuous spectral GMM
    # gs = GridSpec(3, 1, figure=fig)
    # ax1 = fig.add_subplot(gs[0,index])

    continuous_pdfs = plot_kernel_spectrum(Q, cm.kernel, max_x, ax=ax[0], colours=greys, true_freqs=None, scalar=scalar)
    
    # Discontinuous-control spectral GMM
    # ax2 = fig.add_subplot(gs[1,index])
    if true_freqs is not None:
        control_pdfs = plot_kernel_spectrum(Q, dcm.kernel, max_x, ax=ax[1], colours=blues, true_freqs=true_freqs[0], scalar=scalar)
        # Discontinuous-intervention spectral GMM
        intervention_pdfs = plot_kernel_spectrum(Q, dim.kernel, max_x, ax=ax[2], colours=greens, true_freqs=true_freqs[1], scalar=scalar)
   
    else:
        plot_kernel_spectrum(Q, dcm.kernel, max_x, ax=ax[1], colours=blues, true_freqs=None, scalar=scalar)
        # Discontinuous-intervention spectral GMM
        plot_kernel_spectrum(Q, dim.kernel, max_x, ax=ax[2], colours=greens, true_freqs=None, scalar=scalar)
    # return fig, gs
    # fig.suptitle(f"$d$ = {d}",size=30)


#     fig.tight_layout()
#     fig.subplots_adjust(top=0.9)



def plot_posterior_model_spectrum(a, max_x=2.5, padding=0.0, true_freqs=None, ylim=None, lineplot=False, scalar=1.0,
                                  xticks=None, yticks=None, num_samples = 1000, num_f_samples=0, predict_y=False):
    """
    Plots a 4x4 subfigure, which covers the continuous fit on data, continuous GMM spectrum, discontinuous regression, and discontinuous GMM
    :arg
    a : BNQDflow analysis
    Q : number of components in Spectral Mixture Kernel
    cm : BNQDflow continuous model
    dcm : BNQDFlow discontinuous control model
    cim : BNQDflow discontinuous intervention model
    max_x: float - maximum range of x axis on spctral GMM plots.

    """
    sns.set(style='ticks')
    cm = a.continuous_model.model
    dcm = a.discontinuous_model.control_model
    dim = a.discontinuous_model.intervention_model

    fig = plt.figure(constrained_layout=True, figsize=(24, 15))
    gs = GridSpec(8, 3, figure=fig)
    ax1 = fig.add_subplot(gs[:4, :-1])  # Continuous model fit
    ax1.set_title('Continuous model fit', fontsize=30.)  # , x=0.75, y=1.02)

    ax2 = fig.add_subplot(gs[1:3, -1])  # Continuous spectral GMM
    ax3 = fig.add_subplot(gs[4:, :-1])  # Discontinuous model fit
    ax3.set_title('Discontinuous model fit', fontsize=30.0)  # , x=0.75, y=1.02)
    ax4 = fig.add_subplot(gs[4:6, -1])  # Discontinuous-control spectral GMM
    ax5 = fig.add_subplot(gs[6:, -1])  # Discontinuous-intervention spectral GMM

    # colour palettes
    greys = ['#393939', '#575757', '#707070', '#898989', '#a4a4a4', '#bfbfbf']
    blues = ['#367bac', '#3787c0', '#4892c6', '#69a6d0', '#8abbdb', '#95c1de']
    greens = ['#265e52', '#3c7e69', '#599d7e', '#79b895', '#9ed0ae', '#c6e5cc']

    a.continuous_model.plot_regression(ax=ax1, n_samples=num_samples, num_f_samples=num_f_samples, padding=padding, ylim=ylim,
                                       lineplot=lineplot, predict_y=predict_y)
    continuous_pdfs = plot_kernel_spectrum(cm.kernel, max_x, title="Continuous spectral density", ax=ax2, colours=greys,
                         true_freqs=None, scalar=scalar)
    #np.save('continuous_pdfs',continuous_pdfs)
    
    f_samples_list = a.discontinuous_model.plot_regression(ax=ax3, n_samples=1000, num_f_samples=num_f_samples, padding=padding, ylim=ylim,lineplot=lineplot, predict_y=predict_y)
    if xticks is not None:
        ax1.set_xticklabels(xticks)
        ax3.set_xticklabels(xticks)
    if yticks is not None:
        ax1.set_yticklabels(yticks)
        ax3.set_yticklabels(yticks)

    if true_freqs is not None:
        control_pdfs=plot_kernel_spectrum(dcm.kernel, max_x, ax=ax4, title="Spectral density of the control group", colours=blues,
                             true_freqs=true_freqs[0], scalar=scalar)
        intervention_pdfs=plot_kernel_spectrum(dim.kernel, max_x, ax=ax5, title="Spectral density of the intervention group",
                             colours=greens, true_freqs=true_freqs[1], scalar=scalar)
        #np.save('control_pdfs', control_pdfs)
        #np.save('intervention_pdfs', intervention_pdfs)
    else:
        control_pdfs = plot_kernel_spectrum(dcm.kernel, max_x, ax=ax4, title="Control spectral density", colours=blues,
                             scalar=scalar)
        intervention_pdfs=plot_kernel_spectrum(dim.kernel, max_x, ax=ax5, title="Intervention spectral density", colours=greens,
                             scalar=scalar)
        #np.save('control_pdfs', control_pdfs)
        #np.save('intervention_pdfs', intervention_pdfs)
    fig.tight_layout()
    fig.subplots_adjust(top=0.9)

    return f_samples_list

