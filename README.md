
# Spectral-Discontinuity-Design
A Python pipeline that implements Spectral Discontinuity Design (SDD) for Interrupted Time Series, based on gpflow 2.0. 

Quasi-experimental designs allow researchers to determine the effect of a treatment, even when randomized controlled trials are infeasible. 
A prominent example is interrupted time series (ITS) design, in which the effect of an intervention is determined by comparing the extrapolation of a model trained on data acquired up to moment of intervention, with the interpolation by a model trained on data up to the intervention. 
Typical approaches for ITS use (segmented) linear regression, and consequently ignore many of the spectral features of time series data. 

This repository provides the code to apply SDD analysis or recreate the findings in the paper. 
