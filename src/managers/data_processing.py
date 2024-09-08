import pandas as pd
import numpy as np
import logging


class DataProcessor:
    def __init__(self, data):
        self.data = data

    def transform_to_eom(self):
        """
        Transform all data to End of Month (EOM).
        """
        self.data = self.data.resample('M').last()
        self.data.index = self.data.index + pd.offsets.MonthEnd(0)
        logging.info("Data transformed to End of Month")

    def calculate_returns(self, target_columns):
        """
        Calculate 1, 3, and 6 month returns for specified target columns.
        
        :param target_columns: List of column names to calculate returns for
        """
        for col in target_columns:
            if col in self.data.columns:
                self.data[f'{col}_1_mo_return'] = self.data[col].pct_change(periods=1)
                self.data[f'{col}_3_mo_return'] = self.data[col].pct_change(periods=3)
                self.data[f'{col}_6_mo_return'] = self.data[col].pct_change(periods=6)
            else:
                logging.warning(f"Column {col} not found in the data. Skipping return calculation for this column.")
        
        logging.info(f"Calculated returns for available target columns")

    def generate_direction_indicators(self, target_columns):
        """
        Generate direction indicators for return columns.
        
        :param target_columns: List of column names to generate direction indicators for
        """
        for col in target_columns:
            for period in ['1_mo', '3_mo', '6_mo']:
                return_col = f'{col}_{period}_return'
                if return_col in self.data.columns:
                    self.data[f'{return_col}_direction'] = np.where(self.data[return_col] > 0, 1, 0)
                else:
                    logging.warning(f"Column {return_col} not found. Skipping direction indicator generation for this column.")
        
        logging.info("Generated direction indicators for return columns")

    def process_data(self, target_columns, save_csv=False, csv_filename='processed_data.csv'):
        """
        Main method to process the data.
        
        :param target_columns: List of column names to calculate returns for
        :param save_csv: Flag to save the processed data as a CSV file
        :param csv_filename: Name of the CSV file to save (default: 'processed_data.csv')
        :return: Processed DataFrame
        """
        self.transform_to_eom()
        self.calculate_returns(target_columns)
        self.generate_direction_indicators(target_columns)
        
        if save_csv:
            self.save_to_csv(csv_filename)
        
        return self.data

    def save_to_csv(self, filename):
        """
        Save the processed data to a CSV file
        
        :param filename: Name of the CSV file to save
        """
        self.data.to_csv(filename, index=True)
        logging.info(f"Processed data saved to {filename}")

# Example usage:
# Assuming 'all_data' is the output from FREDDataManager
# processor = DataProcessor(all_data)
# target_columns = ['High_Yield', 'Investment_Grade', 'Treasury_3M', 'Treasury_6M', 'Treasury_1Y', 'Treasury_2Y', 'Treasury_3Y', 'Treasury_5Y', 'Treasury_7Y', 'Treasury_10Y', 'Treasury_20Y', 'Treasury_30Y']
# processed_data = processor.process_data(target_columns, save_csv=True, csv_filename='processed_data.csv')
