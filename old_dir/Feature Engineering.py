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
import matplotlib.pyplot as plt

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
        df[str(time) + 'm % change'] = np.asarray(perc_change_list)

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
        df[str(time) + 'm variance'] = np.asarray(variance_list)
        
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
        df[str(time) + 'y z-score'] = np.asarray(z_score_list)
        
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
        
def monthly_fut_return(df, feature, time_lens):
    df[feature] = (df[feature] / 100) + 1
    for time in time_lens:
        return_list = []
        for i, row in df.iterrows():
            if i <= (np.max(df.index) - relativedelta(months=time)) and df.index[0] > df.index[-1]:
                return_list.append((np.prod(df[feature][i+relativedelta(months=+time):i]) - 1) * 100)
            elif i <= (np.max(df.index) - relativedelta(months=time)) and df.index[0] < df.index[-1]:
                return_list.append((np.prod(df[feature][i:i+relativedelta(months=+time)]) - 1) * 100)
            else:
                return_list.append(float('NaN'))
        df['Future ' + str(time) + 'm Return'] = return_list
        
def monthly_prev_return(df, feature, time_lens):
    df[feature] = (df[feature] / 100) + 1
    for time in time_lens:
        return_list = []
        for i, row in df.iterrows():
            if i >= (np.min(df.index) + relativedelta(months=time)) and df.index[0] > df.index[-1]:
                return_list.append((np.prod(df[feature][i:i-relativedelta(months=+time)]) - 1) * 100)
            elif i >= (np.min(df.index) + relativedelta(months=time)) and df.index[0] < df.index[-1]:
                return_list.append((np.prod(df[feature][i-relativedelta(months=+time):i]) - 1) * 100)
            else:
                return_list.append(float('NaN'))
        df['Previous' + str(time) + 'm Return'] = return_list
        
def direction(df, feature):
    df[feature + ' direction'] = [1 if value > 0 else 0 for value in df[feature]]
    
def magnitude(df, feature, mag):
    df[feature + ' magnitude'] = [1 if np.absolute(value) > mag else 0 for value in df[feature]]

"""               
Vix = pd.read_csv('../Data/^VIX.csv')
Vix['Date'] = pd.to_datetime(Vix['Date'])
Vix.set_index('Date', inplace = True)
"""

US_Returns = pd.read_csv('../Data/US IG and US HY Returns.csv')
US_Returns['Value Date'] = pd.to_datetime(US_Returns['Value Date'])
US_Returns.set_index('Value Date', inplace = True)

monthly_prev_return(US_Returns, 'Excess Return', [1])
#direction(US_Returns, 'Future 1m Return')
#magnitude(US_Returns, 'Future 1m Return', 3)






