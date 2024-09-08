# Import necessary packages
import pandas as pd
import numpy as np

# Provides ability to view all columns in PyCharm for validation
desired_width = 320
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns', 100)


# define a fx to calculate monthly returns from daily returns for US IG/HY Datasets
def total_return_from_returns(returns):
    """
    Returns the return between the first and last value of a dataframe; this will be used
    for calculating the EOM data returns for US HY/IG Corporate bond data given daily returns

    :param returns: panda.Series or pandas.DataFrame
    :return:
    """
    return ((returns + 1).prod() - 1) / 100


# Start w/ Data
class DataLoader:
    """
    This class will be used to load data all data sets needed for any type of model. This will read in macro
    datasets like Macro_variables.xlsx, FRED_data.xlsx, ^VIX.csv, CAPE.csv. This will also read in returns
    data for US IG and HY Returns.xlsx and UST Daily Data.xlsx as well as all corresponding data w/ each type of
    maturity of treasury security, seen in the different sheets.

    Attributes:

         | **macro_path** path to the Macro_Variables.xlsx on user's computer system
         | **macro_path_sht** sheet name to be called w/i Macro_Variables.xslx
         | **ret_path** path to the US IG and US HY  Returns.xlsx
         | **HY_ret_path_sht** sheet name to be called w/i US IG and US HY Returns.xslx that is used for
                            HYCorp - HY corporate bonds
         | **IG_ret_path_sht** sheet name to be called w/i US IG and US HY Returns.xlsx that is used for
                            IGCorp - IG Corporate bonds
         | **vix_path** path to the ^VIX.csv on user's computer system
         | **cape_path** path to the CAPE.csv on user's computer system
         | **fred_path** path to the FRED.xlsx on user's computer system
         | **fred_path_sht** sheet name to be called w/i FRED.xlsx
         | **treasury_path** path to the UST Daily Data.xlsx
         | **treasury_path_sht** list of sheets to be read in from UST Daily Data.xlsx
         | **macro_var_data** pandas dataframe that will take in macro_path & macro_path_sht
         | **excess_return_data** pandas dataframe that will take in ret_path & ret_path_sht
         | **vix_data** pandas dataframe that will take in vix_path
         | **cape_data** pandas dataframe that will take in cape_path
         | **treasury_returns

    """

    def __init__(self, macro_path, macro_path_sht, ret_path, hy_ret_path_sht, ig_ret_path_sht, vix_path,
                 cape_path, fred_path, fred_path_sht, treasury_path=None, treasury_path_sht_list=None):
        # Path to CSV on local disk for all files
        self.macro_path = macro_path
        self.macro_path_sht = macro_path_sht
        self.ret_path = ret_path
        self.HY_ret_path_sht = hy_ret_path_sht
        self.IG_ret_path_sht = ig_ret_path_sht
        self.vix_path = vix_path
        self.cape_path = cape_path
        self.fred_path = fred_path
        self.fred_path_sht = fred_path_sht
        self.treasury_path = treasury_path
        self.treasury_path_sht_list = treasury_path_sht_list

        # Intermediate dataframes loaded
        self.macro_var_data = None
        self.HY_excess_return_data = None
        self.IG_excess_return_data = None
        self.excess_return_data = None
        self.vix_data = None
        self.cape_data = None
        self.fred_data = None
        self.UST_data = None
        self.UST1_3_data = None
        self.UST3_5_data = None
        self.UST1_5_data = None
        self.UST5_7_data = None
        self.UST7_10_data = None
        self.UST10_20_data = None
        self.UST20plus_data = None
        self.UST_int_data = None
        self.UST_long_data = None

        # Daily data needed from two macro variable data sets: VIX and Macro_Variables
        self.macro_var_daily_data = None

        # EOM data needed --> create EoM using above fx. and daily data for the same two
        # macro variable data sets: VIX and Macro_Variables
        # Also create EOM data for UST Daily Data.xlsx - Is this necessary?
        self.macro_var_eom_data = None
        self.vix_eom_data = None
        self.UST_eom_data = None
        self.UST1_3_eom_data = None
        self.UST3_5_eom_data = None
        self.UST1_5_eom_data = None
        self.UST5_7_eom_data = None
        self.UST7_10_eom_data = None
        self.UST10_20_eom_data = None
        self.UST20plus_eom_data = None
        self.UST_int_eom_data = None
        self.UST_long_eom_data = None

        # Final dataframes loaded - will be outputs in the end at the .output_csv()
        self.merged_data_eom = None
        self.excess_return_data_eom = None

    def load_data(self):
        """
        Function that loads all necessary data files as defined above for running ML
        or neural nets on Corporate bonds and related data.

        :return: a pandas DataFrame
        """

        # first, deal with Marco_variables.xlsx
        # PLEASE READ!!!!!!!
        # Excel transformations required before loading into notebook. Delete the second row of titles, and then
        # rename the first column of the data - Dates
        self.macro_var_data = pd.read_excel(self.macro_path, self.macro_path_sht)

        # Deal with HY returns data
        self.HY_excess_return_data = pd.read_excel(self.ret_path, self.HY_ret_path_sht)
        self.HY_excess_return_data.drop('Daily Total Return', axis=1, inplace=True)

        # Deal with IG returns data
        self.IG_excess_return_data = pd.read_excel(self.ret_path, self.IG_ret_path_sht)
        self.IG_excess_return_data.drop('Daily Total Return', axis=1, inplace=True)

        # Deal with VIX data
        self.vix_data = pd.read_csv(self.vix_path)
        # Volume column has no data --> all 0's
        self.vix_data.drop('Volume', axis=1, inplace=True)
        # Put VIX in front of each column name to ease the process of reading the macro data set
        self.vix_data.columns = ['Date', 'VIX Open', 'VIX High', 'VIX Low', 'VIX Close', 'VIX Adj Close']

        # deal with CAPE data
        # PLEASE READ!!!!!!!!
        #   Excel transformations required before loading the sheet into this notebook. The date is originally
        #   reported in YYYY.MM - which is not compatible in Python. Therefore, perform the following excel
        #   formula to transform the data into YYYY-MM:
        self.cape_data = pd.read_csv(self.cape_path)
        self.cape_data.drop('Bad Date', axis=1, inplace=True)
        self.cape_data.drop('Intermediate Date', axis=1, inplace=True)

        # deal with FRED data
        # PLEASE READ!!!!!!!
        # Excel transformations required before loading the sheet into this notebook. There is no date column label;
        # therefore name the first column Date
        self.fred_data = pd.read_excel(self.fred_path, self.fred_path_sht)

        # deal with UST Data
        for sht in self.treasury_path_sht_list:
            if sht == 'UST':
                self.UST_data = pd.read_excel(self.treasury_path, sht, skiprows=8)
            elif sht == 'UST1-3':
                self.UST1_3_data = pd.read_excel(self.treasury_path, sht, skiprows=8)
            elif sht == 'UST3-5':
                self.UST3_5_data = pd.read_excel(self.treasury_path, sht, skiprows=8)
            elif sht == 'UST1-5':
                self.UST1_5_data = pd.read_excel(self.treasury_path, sht, skiprows=8)
            elif sht == 'UST5-7':
                self.UST5_7_data = pd.read_excel(self.treasury_path, sht, skiprows=8)
            elif sht == 'UST7-10':
                self.UST7_10_data = pd.read_excel(self.treasury_path, sht, skiprows=8)
            elif sht == 'UST10-20':
                self.UST10_20_data = pd.read_excel(self.treasury_path, sht, skiprows=8)
            elif sht == 'UST 20+':
                self.UST20plus_data = pd.read_excel(self.treasury_path, sht, skiprows=8)
            elif sht == 'UST Int':
                self.UST_int_data = pd.read_excel(self.treasury_path, sht, skiprows=8)
            elif sht == 'UST Long':
                self.UST_long_data = pd.read_excel(self.treasury_path, sht, skiprows=8)

    def preprocess_data(self):
        """
        Function that prepares data to run ML models and possible neural nets. Implementation
        includes dropping columns w/ corrupted/incorrect data; merge raw data into a single
        pandas DataFrame.

        :return: a pandas DataFrame pre-processed and ready for learning
        """

        # List for later use w/ data processing for UST Daily Data
        ust_data_lst = [self.UST_data, self.UST1_3_data, self.UST3_5_data, self.UST1_5_data,
                        self.UST5_7_data, self.UST7_10_data, self.UST10_20_data, self.UST20plus_data,
                        self.UST_int_data, self.UST_long_data]

        """[self.UST_data, self.UST1_3_data, self.UST3_5_data, self.UST1_5_data, self.UST5_7_data,
                    self.UST7_10_data, self.UST10_20_data, self.UST20plus_data, self.UST_int_data,
                    self.UST_long_data]"""

        # Convert date column to a datetime index for all dataframes
        # Once done, set the index for each table equal to that date time object
        self.macro_var_data['Dates'] = pd.to_datetime(self.macro_var_data['Dates'].astype(str),
                                                      format='%Y-%m-%d')
        self.macro_var_data.set_index('Dates', inplace=True)
        self.HY_excess_return_data['Value Date'] = pd.to_datetime(self.HY_excess_return_data['Value Date'].astype(str),
                                                                  format='%Y-%m-%d')
        self.IG_excess_return_data['Value Date'] = pd.to_datetime(self.IG_excess_return_data['Value Date'].astype(str),
                                                                  format='%Y-%m-%d')
        self.HY_excess_return_data.set_index('Value Date', inplace=True)
        self.IG_excess_return_data.set_index('Value Date', inplace=True)
        self.vix_data['Date'] = pd.to_datetime(self.vix_data['Date'].astype(str),
                                               format='%m/%d/%Y')
        self.vix_data.set_index('Date', inplace=True)
        self.cape_data['Date'] = pd.to_datetime(self.cape_data['Date'].astype(str),
                                                format='%Y.%m')
        self.cape_data.set_index('Date', inplace=True)
        self.fred_data['Date'] = pd.to_datetime(self.fred_data['Date'].astype(str),
                                                format='%Y-%m-%d')
        self.fred_data.set_index('Date', inplace=True)

        # convert all UST data to datetime index
        for table in ust_data_lst:
            table['Value Date'] = pd.to_datetime(table['Value Date'].astype(str),
                                                 format='%Y-%m-%d')
            table.set_index('Value Date', inplace=True)

        # Return the macro variable daily data
        self.macro_var_daily_data = pd.merge(self.macro_var_data, self.vix_data, how='outer', right_index=True,
                                             left_index=True)

        # Before we merge the data sets, we need to take away the daily data until
        # we are left with EoM data only.
        # Macro-Variables - daily data --> eom data
        self.macro_var_eom_data = self.macro_var_data.resample('MS').first()

        # VIX - daily data --> eom data
        self.vix_eom_data = self.vix_data.resample('MS').first()

        # Merge all 4 data sets (VIX, CAPE, FRED, and MACRO_VAR's) to create a massive factor dataset for feature
        # engineering.
        self.merged_data_eom = pd.merge(self.vix_eom_data, self.cape_data, how='outer', right_index=True,
                                        left_index=True)
        self.merged_data_eom = pd.merge(self.merged_data_eom, self.macro_var_eom_data, how='outer', right_index=True,
                                        left_index=True)
        self.merged_data_eom = pd.merge(self.merged_data_eom, self.fred_data, how='outer', right_index=True,
                                        left_index=True)

        # Restrict the dataset size by only returning all years >= 1995
        self.macro_var_daily_data = self.macro_var_daily_data['1998-09-01':]
        self.macro_var_daily_data.index.name = 'Date'
        self.merged_data_eom = self.merged_data_eom['1998-09-01':]
        # print(self.merged_data.head(100))

        # RETURN DATA
        # Merge the two data sets into one w/ HY and IG
        self.excess_return_data = pd.merge(self.HY_excess_return_data, self.IG_excess_return_data, how='outer',
                                           right_index=True, left_index=True)
        # Rename columns for ease of calling and visual help
        self.excess_return_data.columns = ['HY Excess Return', 'IG Excess Return']
        # Restrict the data set size like in macro_var_daily_data
        self.excess_return_data = self.excess_return_data['1999-09-01':]
        # Calculate end of month data
        self.excess_return_data_eom = self.excess_return_data.resample('MS').apply(total_return_from_returns)

        # UST DATA
        # Drop the following columns: Return Type, MTD Paydown Return, MTD Currency Return, Current Yield,
        # Average Life, Stripped Yield, Stripped Treasury Spread, Stripped Sovereign Duration, Stripped Spread
        # Duration, Stripped Treasury Duration, Duration (Mod. to Worst), Time to Worst, Excess Return
        for table in ust_data_lst:
            table.drop('Return Type', axis=1, inplace=True)
            table.drop('MTD Paydown Return', axis=1, inplace=True)
            table.drop('MTD Currency Return', axis=1, inplace=True)
            table.drop('Currency', axis=1, inplace=True)
            table.drop('Current Yield', axis=1, inplace=True)
            table.drop('Average Life', axis=1, inplace=True)
            table.drop('Blended Treasury Spread', axis=1, inplace=True)
            table.drop('Stripped Yield', axis=1, inplace=True)
            table.drop('Stripped Treasury Spread', axis=1, inplace=True)
            table.drop('Stripped Sovereign Duration', axis=1, inplace=True)
            table.drop('Stripped Spread Duration', axis=1, inplace=True)
            table.drop('Stripped Treasury Duration', axis=1, inplace=True)
            table.drop('Duration (Mod. to Worst)', axis=1, inplace=True)
            table.drop('Time to Worst', axis=1, inplace=True)
            table.drop('Excess Return', axis=1, inplace=True)

        # restrict the dataset size for the treasury data sets like the above two
        self.UST_data = self.UST_data[:'1999-09-01']
        self.UST1_3_data = self.UST1_3_data[:'1999-09-01']
        self.UST1_5_data = self.UST1_5_data[:'1999-09-01']
        self.UST3_5_data = self.UST3_5_data[:'1999-09-01']
        self.UST5_7_data = self.UST5_7_data[:'1999-09-01']
        self.UST7_10_data = self.UST7_10_data[:'1999-09-01']
        self.UST10_20_data = self.UST10_20_data[:'1999-09-01']
        self.UST20plus_data = self.UST20plus_data[:'1999-09-01']
        self.UST_int_data = self.UST_int_data[:'1999-09-01']
        self.UST_long_data = self.UST_long_data[:'1999-09-01']

        print(self.UST_data.head(10))
        print(self.UST_data.tail(10))

        # create eom data from the treasuries data
        self.UST_eom_data = self.UST_data.resample('MS').agg({'Number Issues (Returns)': ['first'],
                                                              'Returns Modified Duration': ['first'],
                                                              'Daily Total Return': total_return_from_returns,
                                                              'MTD Price Return': ['first'],
                                                              'MTD Coupon Return': ['first'],
                                                              'MTD Total Return': ['first'],
                                                              'Total Return 3 Month': ['first'],
                                                              'Total Return 6 Month': ['first'],
                                                              'YTD Total Return': ['first'],
                                                              'Total Return 12 Month': ['first'],
                                                              'Since Inception Total Return': ['first'],
                                                              'Number Issues (Statistics)': ['first'],
                                                              'Duration (Mod. Adj.)': ['first'],
                                                              'Convexity': ['first'],
                                                              'Coupon': ['first'],
                                                              'Maturity': ['first'],
                                                              'Price': ['first'],
                                                              'Yield to Worst': ['first'],
                                                              'Market Value (MM)': ['first'],
                                                              'Yield to Maturity': ['first'],
                                                              'Blended Spread Duration': ['first'],
                                                              'OAS': ['first'],
                                                              'Amt Outstanding (MM)': ['first']})

        self.UST1_3_eom_data = self.UST1_3_data.resample('MS').agg({'Number Issues (Returns)': ['first'],
                                                                    'Returns Modified Duration': ['first'],
                                                                    'Daily Total Return': total_return_from_returns,
                                                                    'MTD Price Return': ['first'],
                                                                    'MTD Coupon Return': ['first'],
                                                                    'MTD Total Return': ['first'],
                                                                    'Total Return 3 Month': ['first'],
                                                                    'Total Return 6 Month': ['first'],
                                                                    'YTD Total Return': ['first'],
                                                                    'Total Return 12 Month': ['first'],
                                                                    'Since Inception Total Return': ['first'],
                                                                    'Number Issues (Statistics)': ['first'],
                                                                    'Duration (Mod. Adj.)': ['first'],
                                                                    'Convexity': ['first'],
                                                                    'Coupon': ['first'],
                                                                    'Maturity': ['first'],
                                                                    'Price': ['first'],
                                                                    'Yield to Worst': ['first'],
                                                                    'Market Value (MM)': ['first'],
                                                                    'Yield to Maturity': ['first'],
                                                                    'Blended Spread Duration': ['first'],
                                                                    'OAS': ['first'],
                                                                    'Amt Outstanding (MM)': ['first']})

        self.UST1_5_eom_data = self.UST1_5_data.resample('MS').agg({'Number Issues (Returns)': ['first'],
                                                                    'Returns Modified Duration': ['first'],
                                                                    'Daily Total Return': total_return_from_returns,
                                                                    'MTD Price Return': ['first'],
                                                                    'MTD Coupon Return': ['first'],
                                                                    'MTD Total Return': ['first'],
                                                                    'Total Return 3 Month': ['first'],
                                                                    'Total Return 6 Month': ['first'],
                                                                    'YTD Total Return': ['first'],
                                                                    'Total Return 12 Month': ['first'],
                                                                    'Since Inception Total Return': ['first'],
                                                                    'Number Issues (Statistics)': ['first'],
                                                                    'Duration (Mod. Adj.)': ['first'],
                                                                    'Convexity': ['first'],
                                                                    'Coupon': ['first'],
                                                                    'Maturity': ['first'],
                                                                    'Price': ['first'],
                                                                    'Yield to Worst': ['first'],
                                                                    'Market Value (MM)': ['first'],
                                                                    'Yield to Maturity': ['first'],
                                                                    'Blended Spread Duration': ['first'],
                                                                    'OAS': ['first'],
                                                                    'Amt Outstanding (MM)': ['first']})

        self.UST3_5_eom_data = self.UST3_5_data.resample('MS').agg({'Number Issues (Returns)': ['first'],
                                                                    'Returns Modified Duration': ['first'],
                                                                    'Daily Total Return': total_return_from_returns,
                                                                    'MTD Price Return': ['first'],
                                                                    'MTD Coupon Return': ['first'],
                                                                    'MTD Total Return': ['first'],
                                                                    'Total Return 3 Month': ['first'],
                                                                    'Total Return 6 Month': ['first'],
                                                                    'YTD Total Return': ['first'],
                                                                    'Total Return 12 Month': ['first'],
                                                                    'Since Inception Total Return': ['first'],
                                                                    'Number Issues (Statistics)': ['first'],
                                                                    'Duration (Mod. Adj.)': ['first'],
                                                                    'Convexity': ['first'],
                                                                    'Coupon': ['first'],
                                                                    'Maturity': ['first'],
                                                                    'Price': ['first'],
                                                                    'Yield to Worst': ['first'],
                                                                    'Market Value (MM)': ['first'],
                                                                    'Yield to Maturity': ['first'],
                                                                    'Blended Spread Duration': ['first'],
                                                                    'OAS': ['first'],
                                                                    'Amt Outstanding (MM)': ['first']})

        self.UST5_7_eom_data = self.UST5_7_data.resample('MS').agg({'Number Issues (Returns)': ['first'],
                                                                    'Returns Modified Duration': ['first'],
                                                                    'Daily Total Return': total_return_from_returns,
                                                                    'MTD Price Return': ['first'],
                                                                    'MTD Coupon Return': ['first'],
                                                                    'MTD Total Return': ['first'],
                                                                    'Total Return 3 Month': ['first'],
                                                                    'Total Return 6 Month': ['first'],
                                                                    'YTD Total Return': ['first'],
                                                                    'Total Return 12 Month': ['first'],
                                                                    'Since Inception Total Return': ['first'],
                                                                    'Number Issues (Statistics)': ['first'],
                                                                    'Duration (Mod. Adj.)': ['first'],
                                                                    'Convexity': ['first'],
                                                                    'Coupon': ['first'],
                                                                    'Maturity': ['first'],
                                                                    'Price': ['first'],
                                                                    'Yield to Worst': ['first'],
                                                                    'Market Value (MM)': ['first'],
                                                                    'Yield to Maturity': ['first'],
                                                                    'Blended Spread Duration': ['first'],
                                                                    'OAS': ['first'],
                                                                    'Amt Outstanding (MM)': ['first']})

        self.UST7_10_eom_data = self.UST7_10_data.resample('MS').agg({'Number Issues (Returns)': ['first'],
                                                                      'Returns Modified Duration': ['first'],
                                                                      'Daily Total Return': total_return_from_returns,
                                                                      'MTD Price Return': ['first'],
                                                                      'MTD Coupon Return': ['first'],
                                                                      'MTD Total Return': ['first'],
                                                                      'Total Return 3 Month': ['first'],
                                                                      'Total Return 6 Month': ['first'],
                                                                      'YTD Total Return': ['first'],
                                                                      'Total Return 12 Month': ['first'],
                                                                      'Since Inception Total Return': ['first'],
                                                                      'Number Issues (Statistics)': ['first'],
                                                                      'Duration (Mod. Adj.)': ['first'],
                                                                      'Convexity': ['first'],
                                                                      'Coupon': ['first'],
                                                                      'Maturity': ['first'],
                                                                      'Price': ['first'],
                                                                      'Yield to Worst': ['first'],
                                                                      'Market Value (MM)': ['first'],
                                                                      'Yield to Maturity': ['first'],
                                                                      'Blended Spread Duration': ['first'],
                                                                      'OAS': ['first'],
                                                                      'Amt Outstanding (MM)': ['first']})

        self.UST10_20_eom_data = self.UST10_20_data.resample('MS').agg({'Number Issues (Returns)': ['first'],
                                                                        'Returns Modified Duration': ['first'],
                                                                        'Daily Total Return': total_return_from_returns,
                                                                        'MTD Price Return': ['first'],
                                                                        'MTD Coupon Return': ['first'],
                                                                        'MTD Total Return': ['first'],
                                                                        'Total Return 3 Month': ['first'],
                                                                        'Total Return 6 Month': ['first'],
                                                                        'YTD Total Return': ['first'],
                                                                        'Total Return 12 Month': ['first'],
                                                                        'Since Inception Total Return': ['first'],
                                                                        'Number Issues (Statistics)': ['first'],
                                                                        'Duration (Mod. Adj.)': ['first'],
                                                                        'Convexity': ['first'],
                                                                        'Coupon': ['first'],
                                                                        'Maturity': ['first'],
                                                                        'Price': ['first'],
                                                                        'Yield to Worst': ['first'],
                                                                        'Market Value (MM)': ['first'],
                                                                        'Yield to Maturity': ['first'],
                                                                        'Blended Spread Duration': ['first'],
                                                                        'OAS': ['first'],
                                                                        'Amt Outstanding (MM)': ['first']})

        self.UST_int_eom_data = self.UST_int_data.resample('MS').agg({'Number Issues (Returns)': ['first'],
                                                                      'Returns Modified Duration': ['first'],
                                                                      'Daily Total Return': total_return_from_returns,
                                                                      'MTD Price Return': ['first'],
                                                                      'MTD Coupon Return': ['first'],
                                                                      'MTD Total Return': ['first'],
                                                                      'Total Return 3 Month': ['first'],
                                                                      'Total Return 6 Month': ['first'],
                                                                      'YTD Total Return': ['first'],
                                                                      'Total Return 12 Month': ['first'],
                                                                      'Since Inception Total Return': ['first'],
                                                                      'Number Issues (Statistics)': ['first'],
                                                                      'Duration (Mod. Adj.)': ['first'],
                                                                      'Convexity': ['first'],
                                                                      'Coupon': ['first'],
                                                                      'Maturity': ['first'],
                                                                      'Price': ['first'],
                                                                      'Yield to Worst': ['first'],
                                                                      'Market Value (MM)': ['first'],
                                                                      'Yield to Maturity': ['first'],
                                                                      'Blended Spread Duration': ['first'],
                                                                      'OAS': ['first'],
                                                                      'Amt Outstanding (MM)': ['first']})

        self.UST_long_eom_data = self.UST_long_data.resample('MS').agg({'Number Issues (Returns)': ['first'],
                                                                        'Returns Modified Duration': ['first'],
                                                                        'Daily Total Return': total_return_from_returns,
                                                                        'MTD Price Return': ['first'],
                                                                        'MTD Coupon Return': ['first'],
                                                                        'MTD Total Return': ['first'],
                                                                        'Total Return 3 Month': ['first'],
                                                                        'Total Return 6 Month': ['first'],
                                                                        'YTD Total Return': ['first'],
                                                                        'Total Return 12 Month': ['first'],
                                                                        'Since Inception Total Return': ['first'],
                                                                        'Number Issues (Statistics)': ['first'],
                                                                        'Duration (Mod. Adj.)': ['first'],
                                                                        'Convexity': ['first'],
                                                                        'Coupon': ['first'],
                                                                        'Maturity': ['first'],
                                                                        'Price': ['first'],
                                                                        'Yield to Worst': ['first'],
                                                                        'Market Value (MM)': ['first'],
                                                                        'Yield to Maturity': ['first'],
                                                                        'Blended Spread Duration': ['first'],
                                                                        'OAS': ['first'],
                                                                        'Amt Outstanding (MM)': ['first']})

        self.UST20plus_eom_data = self.UST20plus_data.resample('MS').agg({'Number Issues (Returns)': ['first'],
                                                                          'Returns Modified Duration': ['first'],
                                                                          'Daily Total Return': total_return_from_returns,
                                                                          'MTD Price Return': ['first'],
                                                                          'MTD Coupon Return': ['first'],
                                                                          'MTD Total Return': ['first'],
                                                                          'Total Return 3 Month': ['first'],
                                                                          'Total Return 6 Month': ['first'],
                                                                          'YTD Total Return': ['first'],
                                                                          'Total Return 12 Month': ['first'],
                                                                          'Since Inception Total Return': ['first'],
                                                                          'Number Issues (Statistics)': ['first'],
                                                                          'Duration (Mod. Adj.)': ['first'],
                                                                          'Convexity': ['first'],
                                                                          'Coupon': ['first'],
                                                                          'Maturity': ['first'],
                                                                          'Price': ['first'],
                                                                          'Yield to Worst': ['first'],
                                                                          'Market Value (MM)': ['first'],
                                                                          'Yield to Maturity': ['first'],
                                                                          'Blended Spread Duration': ['first'],
                                                                          'OAS': ['first'],
                                                                          'Amt Outstanding (MM)': ['first']})

        print(self.UST_eom_data.head(10))
        print(self.UST_eom_data.tail(10))

    # def output_csv(self):
        """
        This function outputs .csv files to be used in feature.py and model.py to make for quicker
        and easier feature engineering and model development

        :return: 4 different csv files
                    1 - eom Macro variable data from FRED, CAPE, VIX and Macro_Variables
                    2 - excess return data for US IG or HY, whichever inputted into class
                    3 - daily Macro variable data from VIX and Macro_Variables
                    4 - eom return data for US IG or HY, whichever inputted into class
        """
        # Output a csv for merged data
    #    return


# Practice iterations and validate code is outputting the right outputs.
myData = DataLoader(macro_path='Macro_Variables.xlsx', macro_path_sht='Sheet1',
                    ret_path='US IG and US HY Returns.xlsx', hy_ret_path_sht='HYCorp', ig_ret_path_sht='IGCorp',
                    vix_path='^VIX.csv', cape_path='CAPE.csv', fred_path='FRED_Data.xlsx',
                    fred_path_sht='Sheet1', treasury_path='UST Daily Data.xlsx',
                    treasury_path_sht_list=['UST', 'UST1-3', 'UST3-5', 'UST1-5', 'UST5-7', 'UST7-10',
                                            'UST10-20', 'UST 20+', 'UST Int', 'UST Long'])
myData.load_data()
myData.preprocess_data()
# myData.output_csv()
