import numpy as np
import gpflow
import tensorflow as tf
from gpflow.utilities import print_summary, positive
from gpflow.base import Parameter
from gpflow.kernels import Kernel, Sum
from typing import List, Optional, Union
from gpflow.utilities.ops import square_distance
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
f64 = gpflow.utilities.to_default_float
from scipy.fftpack import fft
import math
from scipy.integrate import cumtrapz
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

def SpectralMixture(Q, mixture_weights=None, frequencies=None, lengthscales=None,max_freq=1.0, max_length=1.0,active_dims = None, ytrain=None):
    """
    Spectral Mixture kernel as proposed by Wilson-Adams (2013)
    Currently supports only 1 dimension.
    Parts of code inspired by implementations of:
    - Sami Remes (https://github.com/sremes/nssm-gp/blob/master/nssm_gp/spectral_kernels.py)
    - Srikanth Gadicherla (https://github.com/imsrgadich/gprsm/blob/master/gprsm/spectralmixture.py)
    :arg
    """
    if ytrain is not None:
        emp_frequencies, emp_lengthscales, emp_mixture_weights = initialize_from_emp_spec(Q, ytrain)
        print('Empirical mixture weights: ', emp_mixture_weights)
        print('Empirical frequencies: ', emp_frequencies)
        print('Empirical variances: ', emp_lengthscales)
        lengthscales = 1/np.sqrt(emp_lengthscales)
        print('lengthscales',lengthscales)
        mixture_weights = emp_mixture_weights
        frequencies = emp_frequencies
        

    else:
        if mixture_weights is None:
            mixture_weights = [1.0 for i in range(Q)]
        if frequencies is None:
            frequencies = [((i+1)/Q)* max_freq for i in range(Q)]
        if lengthscales is None:
            #lengthscales = [max_length/Q for _ in range(Q)]
            lengthscales = [max_length for _ in range(Q)]

    components = [SpectralMixtureComponent(i+1, mixture_weights[i], frequencies[i],lengthscales[i], active_dims=active_dims) for i in range(Q)]
    return Sum(components) #if len(components) > 1 else components[0]

def initialize_from_emp_spec(Q, ytrain):
    """:arg
    Function taken entirely from GPyTorch's SM kernel implementation:
    https://gpytorch.readthedocs.io/en/latest/_modules/gpytorch/kernels/spectral_mixture_kernel.html#SpectralMixtureKernel
    """
    N = ytrain.shape[0]
    emp_spect = np.abs(fft(ytrain.flatten())) ** 2 / N
    M = math.floor(N / 2)

    freq1 = np.arange(M + 1)
    freq2 = np.arange(-M + 1, 0)
    freq = np.hstack((freq1, freq2)) / N
    freq = freq[: M + 1]
    emp_spect = emp_spect[: M + 1]
    total_area = np.trapz(emp_spect, freq)
    spec_cdf = np.hstack((np.zeros(1), cumtrapz(emp_spect, freq)))
    spec_cdf = spec_cdf / total_area

    a = np.random.rand(1000, 1)
    p, q = np.histogram(a, spec_cdf)
    bins = np.digitize(a, q)
    slopes = (spec_cdf[bins] - spec_cdf[bins - 1]) / (freq[bins] - freq[bins - 1])
    intercepts = spec_cdf[bins - 1] - slopes * freq[bins - 1]
    inv_spec = (a - intercepts) / slopes

    GMM = GaussianMixture(n_components=Q, covariance_type="diag").fit(inv_spec)
    means = GMM.means_
    varz = GMM.covariances_
    weights = GMM.weights_

    return means.flatten(), varz.flatten(), weights.flatten()



class SpectralMixtureComponent(Kernel):
    """
    Single component of the SM kernel by Wilson-Adams (2013).
    k(x,x') = w * exp(-2 pi^2 * |x-x'| * sigma_q^2 ) * cos(2 pi |x-x'| * mu_q)
    """

    def __init__(self, index, mixture_weight, frequency, lengthscale, active_dims):
        super().__init__(active_dims=active_dims)
        self.index = index

        def logit_transform(min, max):
            a = tf.cast(min, tf.float64)
            b = tf.cast(max, tf.float64)
            affine = tfp.bijectors.AffineScalar(shift=a, scale= (b-a))
            sigmoid = tfp.bijectors.Sigmoid()
            logistic = tfp.bijectors.Chain([affine,sigmoid])
            return logistic
        logistic = logit_transform(0.000001, 100000) # numerical stability
        truncated_frequency = logit_transform(0.000001, 250) # (0, nyquist) (0,100/3)
        self.mixture_weight = gpflow.Parameter(mixture_weight, transform=logistic)
        self.frequency = gpflow.Parameter(frequency, transform=truncated_frequency)
        self.lengthscale = gpflow.Parameter(lengthscale, transform=logistic)
        #self.lengthscale = gpflow.Parameter(lengthscale, transform=positive())
        

        # Sum of Cosine priors 2 cosines (the plots do not use priors currently.)
        #self.frequency.prior = tfd.Normal(f64(frequency), f64(10.))
        #self.mixture_weight.prior = tfd.Gamma(f64(2.), f64(1.0))
        #self.lengthscale.prior = tfd.Gamma(f64(2.), f64(1.))
        
        # Sum of cosine priors 5 cosines
        # self.frequency.prior = tfd.Gamma(f64(2.0), f64(1.))
        # self.mixture_weight.prior = tfd.Gamma(f64(1.0), f64(1.0))
        # self.lengthscale.prior = tfd.Gamma(f64(4.0 ), f64(.2))

        # heart rate priors
        self.lengthscale.prior = tfd.Gamma(f64(8.), f64(.6))
    



    def K(self, X, X2=None):
        if X2 is None:
            X2 = X
        tau_squared = self.scaled_squared_euclid_dist(X,X2)
        exp_term = tf.exp(-2.0 * (np.pi**2)  * tau_squared)

        # Following lines are taken from Sami Remes' implementation (see references above)
        f = tf.expand_dims(X, 1)
        f2 = tf.expand_dims(X2, 0)
        freq = tf.expand_dims(self.frequency, 0)
        freq = tf.expand_dims(freq, 0)
        r = tf.reduce_sum( freq * (f - f2), 2)
        cos_term = tf.cos(r)

        return self.mixture_weight * exp_term * cos_term 

    def K_diag(self, X):
        return tf.fill(tf.shape(X)[:-1], tf.squeeze(self.mixture_weight))


    def scaled_squared_euclid_dist(self, X, X2=None):
        """
        Function to overwrite gpflow.kernels.stationaries
        Returns ||(X - X2ᵀ) / ℓ||² i.e. squared L2-norm.
        """
#         X_scaled = X / self.lengthscale
#         X2_scaled = X2 / self.lengthscale if X2 is not None else X2 / self.lengthscale
#         return square_distance(X_scaled, X2_scaled)
        X = X / self.lengthscale
        Xs = tf.reduce_sum(tf.square(X), axis=1)

        if X2 is None:
            dist = -2 * tf.matmul(X, X, transpose_b=True)
            dist += tf.reshape(Xs, (-1, 1)) + tf.reshape(Xs, (1, -1))
            return dist

        X2 = X2 / self.lengthscale
        X2s = tf.reduce_sum(tf.square(X2), axis=1)
        dist = -2 * tf.matmul(X, X2, transpose_b=True)
        dist += tf.reshape(Xs, (-1, 1)) + tf.reshape(X2s, (1, -1))
        return dist

