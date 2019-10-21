from typing import Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# EDA Reasons & Goals:
#   1) Detection of mistakes
#   2) Checking of assumptions
#   3) Preliminary Selection of appropriate Models
#   4) Determine relationships among explanatory variables
#   5) Direction & rough size of relationships between explanatory
#      & outcome variables

# read in US HY data & quick peek
HY = pd.read_csv('US High Yield.csv')
desired_width = 320
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns', 44)
print(HY.head(10))
print()
print('Columns of HY data are:', HY.columns)
print()

# determine shape of data & look at descriptive stats
print('Shape of US High Yield:', HY.shape)
print()
print(HY.describe(include=[np.number]))
print()
#   289,773 bonds w/ 44 cols (var's) describing each bond
# print(HY.info())
# print()
# categorical attributes of data are:
#   class 2, class 3, class 4, cusip, issue date,
#   prd formatted, ticker,
# numerical attributes are the remaining
# rating_value_factor will not be helpful due to 0 observations
# print(HY.describe(include='all'))

# number of unique values within each numerical attribute
print('# of Unique Values:')
print(HY.nunique())
# class_2 - three sectors
# class_3 - 20 total diff sectors
# class_4 - 65 different smaller sectors
# issue data - issued on 3,779 different dates
# moodys_rating_number - 20 diff categories represented w/i the data
# prd_formatted - 152 of what????? = ASKKKKKKKK!
# sp_rating_number - 23 diff credit ratings
# ticker - 2793 different tickers (bond issuers) in data

HY.index


# UNIVARIATE DISTRIBUTIONS - perform on each component of EDA before we move on to
# multivariate

# CATEGORICAL UNIVARIATE
# start w/ basic aggregate plots for varying classes
# HY['class_2'].value_counts().plot.bar("plt1")
# plt.title('class_2 Frequency')
# plt.show()
# HY['class_3'].value_counts().plot.bar("plt2")
# plt.title('class_3 Frequency')
# plt.show()
# looking at class_4 is not that helpful (too big) - we will only look at class_4
# under each class_2 and class_3
# majority of Bonds are industrial in this dataset; just below 250,000
# financial_institutions and utility make way less than < 50,000
# breaking down by class_3 - top 5 are:
#   consumer_cyclical, comm's, energy, consumer_non_cyclical, basic_industry
#   technology

# look @ industrial class2
# HY[HY['class_2'] == 'INDUSTRIAL']['class_3'].value_counts().plot.bar('plt3')
# plt.title('Industrial class_3 Frequency')
# plt.show()
# results:
#   consumer cyclical, communications, energy are top 3
#   data is NOT CLEAN -> consumer_non_cyclical and consumer_non-cyclical
#       both probably mean the same thing - need to double check
# look @ financial institutions
# HY[HY['class_2'] == 'FINANCIAL_INSTITUTIONS']['class_3'].value_counts().plot.bar('plt4')
# plt.title('Financial Institutions class_3 Frequency')
# plt.show()
# results:
#   finance companies, banking, insurance, REITs are top 4
# look @ utility class
# HY[HY['class_2'] == 'UTILITY']['class_3'].value_counts().plot.bar("plt5")
# plt.title('Utility class_3 Frequency')
# plt.show()
# results: only electric and natural gas and with barely anything under other utilities

# look at the varying class_4 frequencies
# HY[HY['class_2'] == 'INDUSTRIAL']['class_4'].value_counts().plot.bar('plt6')
# plt.title('Industrial class_4 Frequency')
# plt.show()

# HY[HY['class_2'] == 'FINANCIAL_INSTITUTIONS']['class_4'].value_counts().plot.bar('plt7')
# plt.title('Financial Institutions class_4 Frequency')
# plt.show()

# HY[HY['class_2'] == 'UTILITY']['class_4'].value_counts().plot.bar('plt8')
# plt.title('Utility class_4 Frequency')
# plt.show()

# UNIVARIATE NUMERICAL - histograms for all numerical attributes


# start w/ excess return mtd and volatility - iterate through
print()
# print(HY['excess_return_mtd'])
# interest = 'yield_to_worst'
# HY_noNaN = HY[interest].dropna()
# n, bins, patches = plt.hist(x=HY_noNaN, bins='auto',
#                            color='#0504aa', alpha=0.7, rwidth=0.85)
# plt.grid(axis='y', alpha=0.75)
# plt.xlabel(interest)
# plt.ylabel('Frequency')
# plt.title(interest)
# maxfreq = n.max()
# plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
# plt.xlim(-50, 50)
# plt.show()



# BIVARIATE DISTRIBUTION - study relationships between two
# var's using scatterplots

# CORRELATION MATRIX - use this to guide the direction of which multivariate
# distributions are most important to delve into a deeper analysis
#f, ax = plt.subplots(figsize=(16,12))
#corr = HY.corr()
#sns.heatmap(corr,
#            xticklabels=corr.columns.values,
#            yticklabels=corr.columns.values)
#plt.show()

# top 10 bonds for total_return_mtd
#   SYRCF, GARDEN, SCA, DRSDNR, STI, SIXF, HBAN, RMY
#HY_totalReturn = HY.groupby('ticker', as_index=False)['total_return_mtd'].\
#    mean().dropna().sort_values('total_return_mtd', ascending=True)
#print(HY_totalReturn.head(10))
# top least risky bonds - QANAU, KS, ISYSLN, FITB, JORDN, SUNSAG
# top most risky bonds - SYRCF, GARDEN, CNPAU, MUREN, ABK, THMR, KEGX
# top least return bonds - FSTO, NESC, KEGX, IDAR, SOGC, DSL, WM

# BACK TO CORRELATIONS AND BIVARIATE RELATIONSHIPS
# HY_noNaN = HY.dropna()
# var1 = 'coupon'
# var2 = 'oad'
# corr = HY[var1].corr(HY[var2])
# print("Correlation:", corr)
# plt.scatter(HY[var1], HY[var2])
# plt.xlabel(var1)
# plt.ylabel(var2)
# plt.show()

# print("Total percentage:", np.sum(HY['market_value_perc']))
