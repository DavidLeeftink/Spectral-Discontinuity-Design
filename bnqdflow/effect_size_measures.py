import tensorflow as tf
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

from typing import Tuple

from abc import abstractmethod, ABC

from bnqdflow.analyses import Analysis, SimpleAnalysis, PlaceholderAnalysis
from bnqdflow.util import visitor
"""
The code and class structure is written by Melle Starke.
"""

class EffectSizeMeasure(ABC):
    """
    Abstract effect size measure class.
    Allows for the use of different effect size measures, while still being able to share methods and fields.
    """
    def __init__(self):
        self.effect_size = None

    @abstractmethod
    def calculate_effect_size(self, analysis) -> None:
        """
        Abstract effect size measure calculation method.
        This should be used as part of a visitor pattern by all implementations of the EffectSizeMeasure class.
        Implementation of the visitor pattern is done via the bnqdflow.util.visitor decorator.

        :param analysis: Analysis object used as a source for all information required to calculate the effect size.
                         Also used for the visitor patterns implementation.
        :return:
        """
        raise NotImplementedError

class MAE(EffectSizeMeasure):
    """
    Calculates the effect size as the absolute difference between the discontinuous control model and the
    discontinuous intervention model, for the intervention observations.
    d_{ITS}(x) = f^(I)_(x) − f^(C)_(x),  x≥x
    """
    def __init__(self, x_range: Tuple[float, float] = None):
        self.x_range = x_range
        
    @visitor(Analysis)
    def calculate_effect_size(self, analysis):
        raise NotImplementedError("There doesn't exist an implementation for the sharp effect size measure for this "
                                  "analysis: {}".format(analysis.__class__.__name__))

    @visitor(SimpleAnalysis)
    def calculate_effect_size(self, analysis: SimpleAnalysis) -> None:
        """
        Computes the effect size between the control and intevrvention model for all intervention observations
        
        Returns: Mean absolute effect size
        """
        ip        = analysis.intervention_point
        disc_data = analysis.discontinuous_data
        (x_control, y_control), (x_intervention, y_intervention) = disc_data
       
        dcm       = analysis.discontinuous_model.control_model
        dim       = analysis.discontinuous_model.intervention_model
        
        # Means and variances of the two sub-models of the discontinuous model.
        x_intervention = x_intervention.reshape(len(x_intervention),1)
        control_mean, _ = dcm.predict_y(x_intervention)
        interv_mean, _   = dim.predict_y(x_intervention)
  
        causal_effect = np.absolute(np.array(control_mean) - np.array(interv_mean))
        sum_es        = np.sum(causal_effect)
        average_es    = np.mean(causal_effect)
        self.effect_size = (causal_effect, average_es, sum_es)        
        
        

class Sharp(EffectSizeMeasure):
    """
    Sharp effect size measure object.
    Calculates the effect size while assuming there is a discrete separation between the data used by the sub-models of
    the discontinuous model.
    """
    def __init__(self, n_samples: int = 300, n_mc_samples: int = 500, x_range: Tuple[float, float] = None):
        """
        :param n_samples: Number of x-samples used for the effect size distribution.
        :param n_mc_samples: Number of Monte Carlo samples for the BMA density estimate.
        """
        super().__init__()
        self.n_samples = n_samples
        self.n_mc_samples = n_mc_samples
        self.x_range = x_range

    @visitor(Analysis)
    def calculate_effect_size(self, analysis):
        raise NotImplementedError("There doesn't exist an implementation for the sharp effect size measure for this "
                                  "analysis: {}".format(analysis.__class__.__name__))

    @visitor(SimpleAnalysis)
    def calculate_effect_size(self, analysis: SimpleAnalysis) -> None:
        """Computes the effect size at the boundary b. The BMA of the effect
        size is approximated using Monte Carlo sampling and Gaussian kernel
        density estimation.

        Note that this measure of effect size applies only to zeroth-order
        discontinuities, i.e. regression discontinuity.

        :return: Returns a dictionary containing
        - the effect size estimate by the discontinuous model as a pdf
        - the effect size estimate by the discontinuous model as summary
        statistics
        - The BMA effect size estimate.
        - The two-step p-value (i.e. frequentist p-value given the effect size
        distribution by the discontinuous model).
        - The range over which the effect size distribution is given, used for
        plotting.
        - The mean predictions at b.
        - The normalization from standardized effect size to the scale of the
        data.
        """
        ip = analysis.intervention_point

        # Means and variances of the two sub-models of the discontinuous model.
        (m0b, v0b), (m1b, v1b) = analysis.discontinuous_model.predict_y([ip, ip])

        # Mean and standard dev differences. Used to calculate the discontinuous model's effect size estimate.
        disc_mean_diff = np.squeeze(m1b - m0b)  # TODO: why was this swapped around?
        disc_std_diff = tf.sqrt(tf.squeeze(v0b + v1b))

        if disc_mean_diff < 0:
            pval = 1 - stats.norm.cdf(x=0, loc=disc_mean_diff, scale=disc_std_diff)
        else:
            pval = stats.norm.cdf(x=0, loc=disc_mean_diff, scale=disc_std_diff)

        if self.x_range is None:
            xmin, xmax = (np.min([disc_mean_diff - 4 * disc_std_diff, -0.1 * disc_std_diff]),
                          np.max([disc_mean_diff + 4 * disc_std_diff, 0.1 * disc_std_diff]))
        else:
            xmin, xmax = self.x_range

        xrange = np.linspace(xmin, xmax, self.n_samples)

        # Effect size estimate by the discontinuous model as a pdf
        y = stats.norm.pdf(xrange, disc_mean_diff, disc_std_diff)

        samples = np.zeros((self.n_mc_samples))
        nspike = int(np.round(analysis.posterior_model_probabilities()[0] * self.n_mc_samples))
        samples[nspike:] = np.random.normal(loc=disc_mean_diff,
                                            scale=disc_std_diff,
                                            size=(self.n_mc_samples - nspike))

        if not np.isscalar(ip):
            d_bma = None
        else:

            if nspike == self.n_mc_samples:
                # BMA dominated by continuous model
                # Put all mass at xrange closest to b
                d_bma = np.zeros((self.n_samples))
                xdelta = xrange[1] - xrange[0]
                ix = np.argmin((xrange - ip) ** 2)
                d_bma[ix] = 1.0 / xdelta
            elif nspike == 0:
                # BMA dominated by discontinuous model
                d_bma = y
            else:
                # BMA is a mixture
                kde_fit = stats.gaussian_kde(samples,
                                             bw_method='silverman')
                d_bma = kde_fit(xrange)

        self.effect_size = {
            # Estimated Bayesian model average
            'es_bma': d_bma,

            # Estimated effect size by the discontinuous model
            'es_disc': y,

            # Difference in mean and standard deviation of the two sub-models of the discontinuous model.
            'es_disc_stats': (disc_mean_diff, disc_std_diff),

            # Two-step p-value for the discontinuous model.
            'pval': pval,

            # Range over which the effect size distribution is given. Used for plotting
            'es_range': xrange,

            # Mean predictions at the intervention point by the sub-models of the discontinuous model.
            'f(b)': (m0b, m1b),

            # Normalization from the standard effect size to the scale of the data.
            'es_transform': lambda z: z * disc_std_diff + disc_mean_diff
        }

    @visitor(PlaceholderAnalysis)
    def calculate_effect_size(self, analysis: PlaceholderAnalysis) -> None:
        print("There doesn't exist an implementation for Sharp.calculate_effect_size() for {}"
              .format(analysis.__class__.__name__))
        return None

    def plot_bma(self):
        plt.title("BMA effect size")
        x_range = self.effect_size['es_range']
        plt.plot(x_range, self.effect_size['es_bma'], label='BMA')
        plt.plot(x_range, self.effect_size['es_disc'], label='Discontinuous effect size estimate')


class FuzzyEffectSize(EffectSizeMeasure):

    def __init__(self):
        super().__init__()

    def calculate_effect_size(self, analysis) -> None:
        pass
