# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 18:46:16 2019

@author: C20Noah.Miller
"""

import pandas as pd
import numpy as np
from datetime import datetime
from datetime import timedelta
from dateutil.relativedelta import *
from scipy.stats import zscore

def perc_change(df, feature, time_lens):
    """
    calculates percentage change of a feature from a dataframe 
    over certain periods of months and appends them to the dataframe
    """
    for time in time_lens:
        perc_change_list = []
        for i, row in df.iterrows():
            try:
                perc_change_list.append((df[feature][i] - df[feature][i-relativedelta(months=+time)]) 
                / df[feature][i-relativedelta(months=+time)])
            except KeyError:
                perc_change_list.append(float('NaN'))
        df[time, 'm % change'] = np.asarray(perc_change_list)

def variance(df, feature, time_lens):
    """
    calculates the variance of a feature from a dataframe over
    certain periods of months and appends them to the dataframe
    """
    for time in time_lens:
        variance_list = []
        for i, row in df.iterrows():
            if (i - relativedelta(months=+time)) >= df.index[0]:
                variance_list.append(np.var(df[feature][i-relativedelta(months=+time):i]))
            else:
                variance_list.append(float('NaN'))
        df[time, 'm variance'] = np.asarray(variance_list)
        
def z_score(df, feature, time_lens):
    """
    Calculates the z-score for a certain feature from a dataframe
    over certain periods of years and appends them to the dataframe
    """  
    for time in time_lens:
        z_score_list = []
        for i, row in df.iterrows():
            if (i - relativedelta(years=+time)) >= df.index[0]:
                z_score_list.append(zscore(df[feature][i-relativedelta(years=+time):i])[-1])
            else:
                z_score_list.append(float('NaN'))
        df[time, 'y z-score'] = np.asarray(z_score_list)
        
def expon_weight_move_ave(df, feature, time_lens):
    """
    Calculates the exponentially weighted moving average for 
    a certain feature from a dataframe over certain periods of
    months and appends them to the dataframe
    """
    for time in time_lens: 
        sma = df[feature].rolling(window=(time * 21), min_periods=(time * 21)).mean()[:time * 21]
        rest = df[feature][(time * 21):]
        ewma = pd.concat([sma, rest]).ewm(span = (time*21), adjust = False).mean()
        df[time, 'm ewma'] = ewma

                
Vix = pd.read_csv('../Data/^VIX.csv')
Vix['Date'] = pd.to_datetime(Vix['Date'])
Vix.set_index('Date', inplace = True)

perc_change(Vix, 'Adj Close', [1,3,6])
