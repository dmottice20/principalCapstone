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
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

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
        df[str(time) + 'm ewma'] = ewma
        
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



# MODELLLLLLLLL

# daily return data
return_data = pd.read_csv('excess_return_data.csv')
return_data['Value Date'] = pd.to_datetime(return_data['Value Date'])
return_data.set_index('Value Date', inplace=True)

# Daily feature data
daily_data = pd.read_csv('macro_var_daily.csv')
daily_data['Date'] = pd.to_datetime(daily_data['Date'])
daily_data.set_index('Date', inplace=True)

# feature engineer this bitch
variance(daily_data, 'VIX Adj Close', [1, 3])
expon_weight_move_ave(daily_data, '.BAA10YB Index', [3, 6, 12])

# print(daily_data.tail(10))

# drop everything else
featured_daily_data = daily_data[['1m variance', '3m variance', '3m ewma', '6m ewma', '12m ewma', 'SPX Index - 30Day RSI']]
# print(featured_daily_data.tail(20))

# eom only
featured_daily_data_eom = featured_daily_data.resample('MS').last()
# print(featured_daily_data_eom.tail(20))

# return fe
monthly_fut_return(return_data, 'HY Excess Return', [1])
return_data = return_data[['Future 1m Return']]
return_data.columns = ['fut_1m_return']
return_data_eom = return_data.resample('MS').last()
direction(return_data_eom, 'fut_1m_return')
return_data_eom = return_data_eom[['fut_1m_return direction']]
# print(return_data_eom.head(20))

# merege then drop na
merged_data_eom = pd.merge(featured_daily_data_eom, return_data_eom, how='outer', right_index=True, left_index=True)
merged_data_eom = merged_data_eom.dropna()


def repeat_value_and_create_df(df, repeat_val, test_X, test_y):
    '''
    Function reapats a value and append to dataFrame.

    :param df:          Original dataFrame
    :param repeat_val:  Value to repeat
    :param test_X:      Input for test data
    :param test_y:      Output for test data
    :return:            Merged dataFrame
    '''
    df = pd.concat([df, pd.DataFrame(np.reshape(np.repeat(repeat_val, test_y.shape[0]),
                                                [test_X.shape[1], -1], order='C'),
                                     columns=test_y.index, index=test_X.columns)], axis=1)
    return df


def huber_approx_obj(y_true, y_pred):
    '''
    Function that calculates huber approximation.

    :param y_true:      True output response
    :param y_pred:      Predicted response
    :return:            Output from huber approximation
    '''
    d = y_pred - y_true  # remove .get_labels() for sklearn
    h = 1  # h is delta in the graphic
    scale = 1 + (d / h) ** 2
    scale_sqrt = np.sqrt(scale)
    grad = d / scale_sqrt
    hess = 1 / scale / scale_sqrt
    return grad, hess


def train_test_split_USAFA(input_data, response_data):
    """

    :param input_data:
    :param response_data:
    :return:
    """
    train_X = input_data[:'2010-01-01']
    test_X = input_data['2010-01-01':]
    train_y = response_data[:'2010-01-01']
    test_y = response_data['2010-01-01':]

    return train_X, test_X, train_y, test_y


# myModel = PredictionModel(model_type='Lasso', ret='fut_1m_return')
train_X, test_X, train_y, test_y = train_test_split_USAFA(input_data=merged_data_eom.drop(['fut_1m_return direction'], axis = 1),
                                                          response_data=merged_data_eom['fut_1m_return direction'])
# myModel.create_prediction_model(input_data=)
my_model = XGBClassifier(learning_rate =0.1,
                         n_estimators=1000,
                         max_depth=5,
                         min_child_weight=1,
                         gamma=0,
                         subsample=0.8,
                         colsample_bytree=0.8,
                         objective= 'binary:logistic',
                         nthread=4,
                         scale_pos_weight=1
                         )
my_model.fit(train_X, train_y)
y_pred = my_model.predict(test_X)
predictions = [round(value) for value in y_pred]
accuracy = accuracy_score(test_y, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

param_test3 = {
 'gamma':[i/10.0 for i in range(0,5)]
}
gsearch3 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=140, max_depth=4,
 min_child_weight=6, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 
 param_grid = param_test3, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch3.fit(train_X,train_y)
gsearch3.grid_scores_, gsearch3.best_params_, gsearch3.best_score_


