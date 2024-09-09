import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TimeSeriesCrossValidator:
    def __init__(self, data):
        self.data = data

    def rolling_window_split(self, initial_train_size, step_size, test_size):
        """
        Perform rolling window cross-validation.
        
        :param initial_train_size: Initial size of the training set (in days)
        :param step_size: Number of days to move forward for each split
        :param test_size: Size of the test set (in days)
        :return: Generator of (train_index, test_index) tuples
        """
        total_size = len(self.data)
        for train_end in range(initial_train_size, total_size - test_size, step_size):
            train_start = 0
            train_end = train_end
            test_start = train_end
            test_end = test_start + test_size

            if test_end > total_size:
                break

            train_index = self.data.index[train_start:train_end]
            test_index = self.data.index[test_start:test_end]

            yield train_index, test_index

    def sliding_window_split(self, window_size, step_size, test_size):
        """
        Perform sliding window cross-validation.
        
        :param window_size: Size of the training window (in days)
        :param step_size: Number of days to move forward for each split
        :param test_size: Size of the test set (in days)
        :return: Generator of (train_index, test_index) tuples
        """
        total_size = len(self.data)
        for window_start in range(0, total_size - window_size - test_size + 1, step_size):
            train_start = window_start
            train_end = window_start + window_size
            test_start = train_end
            test_end = test_start + test_size

            if test_end > total_size:
                break

            train_index = self.data.index[train_start:train_end]
            test_index = self.data.index[test_start:test_end]

            yield train_index, test_index

    def perform_cross_validation(self, method='rolling', **kwargs):
        """
        Perform cross-validation using the specified method.
        
        :param method: 'rolling' or 'sliding'
        :param kwargs: Additional arguments for the specific method
        :return: List of (train, test) dataframe tuples
        """
        if method == 'rolling':
            split_generator = self.rolling_window_split(**kwargs)
        elif method == 'sliding':
            split_generator = self.sliding_window_split(**kwargs)
        else:
            raise ValueError("Invalid method. Choose 'rolling' or 'sliding'.")

        splits = []
        for train_index, test_index in split_generator:
            train_data = self.data.loc[train_index]
            test_data = self.data.loc[test_index]
            splits.append((train_data, test_data))

        logging.info(f"Created {len(splits)} splits using {method} window method")
        return splits

# Example usage:
# cv = TimeSeriesCrossValidator(processed_data)
# 
# # For rolling window
# rolling_splits = cv.perform_cross_validation(method='rolling', initial_train_size=365*5, step_size=30, test_size=30)
# 
# # For sliding window
# sliding_splits = cv.perform_cross_validation(method='sliding', window_size=365*5, step_size=30, test_size=30)
