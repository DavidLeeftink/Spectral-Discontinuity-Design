# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 11:40:22 2020

@author: Max
"""

import GPy
import numpy as np
import matplotlib.pyplot as plt
import warnings; warnings.simplefilter('ignore')

import datetime as dt

import pandas as pd
import importlib
import sys
sys.path.append('../')
import BNQD
importlib.reload(BNQD)

print("GPy version:      {}".format(GPy.__version__))
print("BNQD version:     {}".format(BNQD.__version__))

plt.close('all')

df_name = 'Datasets/Covid 19 cases/full_data.csv'
# data per 14-03-2020 from https://ourworldindata.org/coronavirus-source-data

df = pd.read_csv(df_name)

print(df.columns)

regions = ['Netherlands', 'China', 'Italy', 'Worldwide', 'United States']

m = len(regions)

for item in ['new_cases', 'new_deaths', 'total_cases', 'total_deaths']:

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15,12), sharex=True)
    
    dates = pd.to_datetime(df.loc[df['location'] == 'Worldwide']['date']).dt.date.unique().tolist()
    
    T = len(dates)
    
    df_ww = df.loc[df['location'] == 'Worldwide']
    
    
    for region in regions:
        
        df_region = df.loc[df['location'] == region]   
        res_subset = pd.merge(df_ww, df_region, how='left', on='date')
        
        vals = res_subset[item + '_y']
        axes[0].plot(dates, vals, label=region)
        
        if region == 'Netherlands':
            axes[1].plot(dates, vals)
    
    axes[0].set_ylabel(item.replace('_', ' '))
    axes[0].axvline(x = pd.Timestamp('2020-01-23'), ls=':', c='k', label='Wuhan lockdown')
    axes[0].legend()
    
    
    fig.autofmt_xdate()
    plt.show()