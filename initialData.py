# Import necessary packages
import pandas as pd
import numpy as np

desired_width = 320
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns', 100)


# Start w/ Data
class DataLoader:
    """
    This class will be used to load data for US High Yield Corporate bond data in .csv format.
    Implementation of the DataLoader class comes from the pandas package in Python. This class
    assumes we will need/use all of the following datasets: VIX, CAPE, FRED, Macro_variables,
    UG IG and HY Returns. In essence, this class is used for Corporate bond datasets.

    Attributes:

         | **merged_data:** DataFrame w/ factor/return data
         | **eom_data:** DataFrame w/ eom data from Macro_variables and Fred csv

    """

    def __init__(self, macroPath='Macro_Variables.xlsx', macroPathSht='Sheet1'
                 , retPath='US IG and US HY Returns.xlsx', retPathSht='HYCorp',
                 vixPath='^VIX.csv', capePath='CAPE.csv', fredPath='FRED_Data.xlsx',
                 fredPathSht='Sheet1'):
        # Path to CSV on local disk for all files
        self.macroPath = macroPath
        self.macroPathSht = macroPathSht
        self.retPath = retPath
        self.retPathSht = retPathSht
        self.vixPath = vixPath
        self.capePath = capePath
        self.fredPath = fredPath
        self.fredPathSht = fredPathSht

        # Intermediate dataframes loaded
        self.Macro_Var_data = None
        self.return_data = None
        self.VIX_data = None
        self.CAPE_data = None
        self.FRED_data = None

        # Final dataframes loaded
        self.merged_data = None
        self.target_data = None

    def load_data(self):
        """
        Function that loads all necessary data files as defined above for running ML
        or neural nets on Corporate bonds and related data.

        :return: a pandas DataFrame
        """

        # first, deal with Marco_variables.xlsx
        self.Macro_Var_data = pd.read_excel(self.macroPath, self.macroPathSht, header=1)
        # Have to manually name columns due to poor db design by FRED in the csv
        self.Macro_Var_data.columns = ['Dates', 'USGG2YR Index', 'USGG10YR Index', 'USGG30YR Index',
                                       'VIX Index', 'MOVE Index', 'AAIIBULL Index',
                                       'AAIIBEAR Index', 'SPX Index - 14Day RSI',
                                       'SPX Index - 30Day RSI', '.TED G Index',
                                       '.BAA10YB Index', 'BFCIUS Index',
                                       'L98TRUU Index', 'SPX Index - Last Measure']

        # Deal with returns data
        self.return_data = pd.read_excel(self.retPath, self.retPathSht, header=1)
        self.return_data.columns = ['Value Date', 'HY Daily Total Return', 'HY Excess Return']
        # print("Pre-preprocessed factor data:")
        # print(self.raw_data_1.head(10))
        # print("Pre-preprocessed return data:")
        # print(self.raw_data_2.head(10))

        # deal with VIX data
        self.VIX_data = pd.read_csv(self.vixPath)
        self.VIX_data.columns = ['Date', 'VIX Open', 'VIX High', 'VIX Low', 'VIX Close', 'VIX Adj Close', 'Volume']
        self.VIX_data.drop('Volume', axis=1, inplace=True)

        # deal with CAPE data
        self.CAPE_data = pd.read_csv(self.capePath)
        self.CAPE_data.drop('Shit_interm', axis=1, inplace=True)
        self.CAPE_data.drop('Date_interm', axis=1, inplace=True)

        # deal with FRED data
        self.FRED_data = pd.read_excel(self.fredPath, self.fredPathSht, header=2)
        self.FRED_data.columns = ['Date', 'RPI', 'W875RX1', 'DPCERA3M0865BEA', 'CMRMTSPL',
                                  'INDPRO', 'IPFPNSS', 'IPFINAL', 'IPCONGD',
                                  'IPDCONGD', 'IPNCONGD', 'IPBUSEQ', 'IPMAT',
                                  'IPDMAT', 'IPNMAT', 'IPMANSICS', 'IPB51222S',
                                  'IPFUELS', 'CUMFNS', 'CUMFNS', 'CLF160V',
                                  'CE160V', 'UNRATE', 'UNEMPMEAN', 'UEMPLT5',
                                  'UEMP5TO14', 'UEMP15OV', 'UEMPT15T26',
                                  'UEMP27OV', 'PAYEMS', 'USGOOD', 'CES1021000001',
                                  'USCONS', 'MANEMP', 'DMANEMP', 'SRVPD', 'USWTRADE',
                                  'USTRADE', 'USFIRE', 'USGOVT', 'CES0600000007',
                                  'AWOTMAN', 'HOUST', 'HOUSENTE', 'HOUSTMW', 'HOUSTS',
                                  'HOUSTW', 'ISRATIO', 'M1SL', 'M2SL', 'M2REAL', 'AMBSL',
                                  'BUSLOANS', 'REALLN', 'WPSFD49207', 'WPSFD49502', 'WPSID61',
                                  'WPSID62', 'MZMSL', 'DTCOLNVHFNM', 'DTCTHFNM', 'INVEST',
                                  'ism_man_empl', 'ism_man_neworders', 'ism_man_pmi',
                                  'ism_man_prices']

    def preprocess_data(self):
        """
        Function that prepares data to run ML models and possible neural nets. Implementation
        includes dropping columns w/ corrupted/incorrect data; merge raw data into a single
        pandas DataFrame.

        :return: a pandas DataFrame pre-processed and ready for learning
        """

        # Convert date column to a datetime index for all dataframes
        # Once done, set the index for each table equal to that date time object
        self.Macro_Var_data['Dates'] = pd.to_datetime(self.Macro_Var_data['Dates'].astype(str),
                                                      format='%Y-%m-%d')
        self.Macro_Var_data.set_index('Dates', inplace=True)
        self.return_data['Value Date'] = pd.to_datetime(self.return_data['Value Date'].astype(str),
                                                        format='%Y-%m-%d')
        self.return_data.set_index('Value Date', inplace=True)
        self.VIX_data['Date'] = pd.to_datetime(self.VIX_data['Date'].astype(str),
                                               format='%Y-%m-%d')
        self.VIX_data.set_index('Date', inplace=True)
        self.CAPE_data['Date'] = pd.to_datetime(self.CAPE_data['Date'].astype(str),
                                                format='%Y.%m')
        self.CAPE_data.set_index('Date', inplace=True)
        self.FRED_data['Date'] = pd.to_datetime(self.FRED_data['Date'].astype(str),
                                                format='%Y-%m-%d')
        self.FRED_data.set_index('Date', inplace=True)

        # Merge data on date column to a make a merged data frame
        # self.merged_data = pd.merge(self.Macro_Var_data, self.return_data, right_on='Value Date',
        #                            left_on='Dates')
        # print(self.merged_data.columns)
        # print(self.merged_data.head(100))

        # Merge all 4 data sets (VIX, CAPE, FRED, and MACRO_VAR's) to create a massive factor dataset for feature
        # engineering.
        self.merged_data = pd.merge(self.VIX_data, self.CAPE_data, how='outer', right_index=True, left_index=True)
        self.merged_data = pd.merge(self.merged_data, self.Macro_Var_data, how='outer', right_index=True,
                                    left_index=True)
        self.merged_data = pd.merge(self.merged_data, self.FRED_data, how='outer', right_index=True, left_index=True)
        # print(self.merged_data.columns)
        # print(self.merged_data.head(100))

        # Restrict the dataset size by only returning all years >= 1995
        self.merged_data = self.merged_data['1998-09-01':]
        print(self.merged_data.head(100))

        # RETURN DATA

        # Convert daily to monthly
        self.target_data = self.return_data

    def output_csv(self):
        """
        This function outputs a .csv file to be used in future to make for quicker and easier feature
        engineering and model development

        :return: 2 different csv files
                    1 - factor/feature data (MACRO)
                    2 - target data
        """
        # Output a csv for merged data
        return self.merged_data.to_csv('merged_data.csv'), self.target_data.to_csv('target_data.csv')


myData = DataLoader()
myData.load_data()
myData.preprocess_data()
myData.output_csv()

