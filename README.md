
# Spectral Discontinuity Design
A Python pipeline that implements [Spectral Discontinuity Design (SDD) for Interrupted Time Series](http://proceedings.mlr.press/v136/leeftink20a/leeftink20a.pdf), based on [gpflow 2.0](https://gpflow.readthedocs.io/en/master/). 

Quasi-experimental designs allow researchers to determine the effect of a treatment, even when randomized controlled trials are infeasible. 
A prominent example is interrupted time series (ITS) design, in which the effect of an intervention is determined by comparing the extrapolation of a model trained on data acquired up to moment of intervention, with the interpolation by a model trained on data up to the intervention. 
Typical approaches for ITS use (segmented) linear regression, and consequently ignore many of the spectral features of time series data. 
In this repository, we show a Bayesian nonparametric approach to ITS, that uses Gaussian process regression and the spectral mixture kernel. The method is demonstrated in simulation as well as to determine the causal effect of Kundalini yoga meditation on heart rate oscillations. 
