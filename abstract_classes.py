import pandas as pd
from abc import ABC, abstractmethod

class DataLoader(ABC):
    '''
    Class that loads data from local CSVs. Implementation of
    DataLoader abstract class from the quant_pipeline package.

    Attributes:

        | **merged_data:** Dataframe with factor/return data

        | **eom_data:** Dataframe with end of month data observations
    '''

    @abstractmethod
    def load_data(self):
        '''
        Function that loads data out of csv.
        '''
        self.raw_data = pd.read_csv(self)

    @abstractmethod
    def preprocess_data(self):
        '''
        Function that changes Date column to datetime and gets only end of
        month data observations

        NOTE: I'm doing this because datasets like VIX need to processed like this
        '''
        self.raw_data["Date"] = pd.to_datetime(raw_data["Date"])
        cols = [col for col in self.raw_data.columns if col not in ["Date","Close", "Adj Close"]]
        self.raw_data = self.raw_data[cols]


        # Gets rid of observations not the close of the month
        ####### Big problem this does not always work because there's no trades on weekends
        from pandas.tseries.offsets import MonthEnd
        for row in self.raw_data.itertuples():
            if row.Date != (row.Date + MonthEnd(1)):
                self.raw_data.drop(row)



class FeatureEngineering(ABC):
    '''
    Abstract base class for Feature Engineering.
    '''

    @abstractmethod
    def input_feature_engineering(self):
        '''
        Function where all input feature generation should occur. Our definition
        of input feature is any input into a model that is not raw data.
        '''
        pass

    @abstractmethod
    def output_feature_engineering(self):
        '''
        Function where all output feature generation should occur. Our
        definition of output feature is any value that is related to the output
        of a model that is not raw data.
        '''
        pass

class Model(ABC):
    '''
    Abstract base class for Modeling (generation of asset level signals).
    '''

    @abstractmethod
    def prediction(self):
        '''
        Function where model is implemented. The model takes inputs from feature
        generation and produces a signal.
        '''
        pass

    @abstractmethod
    def policy_generation(self):
        '''
        Function that generates a policy based on signal from prediction (ex.
        converts predicted porfolio returns to portfolio weights). Not all
        strategies make use of this function.
        '''
        pass

    @abstractmethod
    def asset_scoring(self):
        '''
        Function that converts policy to asset score (ex. converts portfolio
        weights to stock scores). Not all strategies make use of this function.
        '''
        pass
