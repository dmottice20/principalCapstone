import pandas as pd
from fredapi import Fred
from datetime import datetime
import logging
import json


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FREDDataManager:
    def __init__(self, api_key):
        self.fred = Fred(api_key=api_key)
        
    def get_data(self, series_id, start_date):
        """
        Fetch data for a given series from FRED API
        
        :param series_id: FRED series ID
        :param start_date: Start date for data retrieval (format: 'YYYY-MM-DD')
        :return: pandas DataFrame with the requested data
        """
        end_date = datetime.now().strftime('%Y-%m-%d')
        data = self.fred.get_series(series_id, observation_start=start_date, observation_end=end_date)
        df = pd.DataFrame(data, columns=[series_id])
        df.index.name = 'Date'
        return df
    
    def get_all_data(self, series_ids, start_date, save_csv=False, csv_filename='fred_data.csv'):
        """
        Fetch data for multiple series from FRED API and merge them
        
        :param series_ids: List of FRED series IDs
        :param start_date: Start date for data retrieval (format: 'YYYY-MM-DD')
        :param save_csv: Flag to save the data as a CSV file
        :param csv_filename: Name of the CSV file to save (default: 'fred_data.csv')
        :return: pandas DataFrame with all requested data merged
        """

        logging.info("Checking the earliest dates for each series...")
        earliest_dates = {}
        for series_id in series_ids:
            try:    
                data = self.fred.get_series(series_id)
                earliest_dates[series_id] = data.index[0]
            except Exception as e:
                logging.error(f"Error fetching data for {series_id}: {e}")
                earliest_dates[series_id] = None
    
        # Find the maximum earliest date
        max_earliest_date = max(max(earliest_dates.values()), datetime.strptime(start_date, '%Y-%m-%d').date())
        
        # Set the start date to the maximum earliest date
        start_date = max_earliest_date
        logging.info(f"Adjusted start date to {start_date}")

        logging.info("Loading data from FRED...")
        dfs = []
        for series_id in series_ids:
            df = self.get_data(series_id, start_date)
            dfs.append(df)
        
        merged_df = pd.concat(dfs, axis=1)
        logging.info("Data loading and merging completed")

        if save_csv:
            self.save_to_csv(merged_df, csv_filename)

        return merged_df

    def save_to_csv(self, data, filename):
        """
        Save the data to a CSV file
        
        :param data: pandas DataFrame to save
        :param filename: Name of the CSV file to save
        """
        data.to_csv(filename, index=True)  # Ensure index (Date) is included
        logging.info(f"Data saved to {filename}")

# Example usage:
# api_key = 'b7bc70213173b5217454a063e86b72ce'
# start_date = '1998-01-01'
# # series_ids = ['GDP', 'UNRATE', 'CPIAUCSL', 'FEDFUtoucNDS']  # Add all series IDs from merged_data.csv
# series_ids = [value for value in json.load(open('src/managers/data/fred-tickers.json')).values() if value is not None]
# fred_manager = FREDDataManager(api_key)
# all_data = fred_manager.get_all_data(series_ids, start_date, save_csv=True, csv_filename='fred_data_output.csv')
# print(all_data.describe())
# print("Shape of the data: ", all_data.shape)

    # // "ISM Manufacturing Employment Index": "NAPMEI",
    # // "ISM Manufacturing New Orders Index": "NAPMNOI",
    # // "ISM Manufacturing PMI": "NAPMPMI",
    # // "ISM Manufacturing Prices Index": null

