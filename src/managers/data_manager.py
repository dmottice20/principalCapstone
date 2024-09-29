import pandas as pd
from fredapi import Fred
from datetime import datetime
import logging
import yfinance as yf


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
    
    def get_all_data(self, series_ids, start_date):
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
    
        # Find the maximum earliest date and the ticker associated.
        max_earliest_date = max(max(earliest_dates.values()), datetime.strptime(start_date, '%Y-%m-%d').date())
        ticker_earliest_date = max(earliest_dates, key=earliest_dates.get)
        logging.info(f"Earliest date is {max_earliest_date} for {ticker_earliest_date}")

        logging.info("Loading data from FRED...")
        dfs = []
        for series_id in series_ids:
            df = self.get_data(series_id, max_earliest_date)
            dfs.append(df)
        
        merged_df = pd.concat(dfs, axis=1)
        logging.info("FRED data loaded")

        merged_df.to_csv("src/managers/data/fred_data.csv", index=True)

        return merged_df


class YahooDataManager:
    def __init__(self):
        pass

    def get_data(self, ticker):
        """
        Fetch data for a given ticker from Yahoo Finance API
        
        :param ticker: Yahoo Finance ticker symbol
        :param start_date: Start date for data retrieval (format: 'YYYY-MM-DD')
        :return: pandas DataFrame with the requested data
        """
        data = yf.download(ticker, period='max', interval='1d')
        data.index.name = 'Date'
        close_data = data["Close"].to_frame()
        # rename the column to the ticker symbol
        close_data.columns = [ticker]
        close_data.to_csv("src/managers/data/yahoo_data.csv", index=True)
        return close_data
    

class DataFusion:
    def __init__(self, fred_data, yahoo_data):
        self.fred_data = fred_data
        self.yahoo_data = yahoo_data

    def fuse_data(self):
        """
        Merge data from FRED and Yahoo Finance by finding the earliest date on both datasets
        and using the max earliest date as the start date for the merged data

        :return: Merged pandas DataFrame with data from FRED and Yahoo Finance
        """
        # Find common date range.
        common_start_date = max(self.fred_data.index.min(), self.yahoo_data.index.min())
        common_end_date = min(self.fred_data.index.max(), self.yahoo_data.index.max())
        date_range = pd.date_range(start=common_start_date, end=common_end_date)

        # Reindex data to the common date range
        self.fred_data, self.yahoo_data = self.fred_data.reindex(date_range), self.yahoo_data.reindex(date_range)

        # Then, fuse the data.
        return pd.concat([self.fred_data, self.yahoo_data], axis=1)
    
    def save_to_csv(self, data, filename):
        """
        Save the data to a CSV file
        
        :param data: pandas DataFrame to save
        :param filename: Name of the CSV file to save
        """
        data.to_csv(filename, index=True)  # Ensure index (Date) is included
        logging.info(f"Data saved to {filename}")


# # Example usage:
# api_key = 'b7bc70213173b5217454a063e86b72ce'
# start_date = '1998-01-01'

# yahoo = YahooDataManager()
# data = yahoo.get_data('SPY')
# # series_ids = ['GDP', 'UNRATE', 'CPIAUCSL', 'FEDFUtoucNDS']  # Add all series IDs from merged_data.csv
# series_ids = [value for value in json.load(open('src/managers/data/fred-tickers.json')).values() if value is not None]
# fred_manager = FREDDataManager(api_key)
# all_data = fred_manager.get_all_data(series_ids, start_date)

# fusion = DataFusion(all_data, data)
# data = fusion.fuse_data()
# fusion.save_to_csv(data, 'src/managers/data/merged_data.csv')
# # Check how many nulls are in the data by each column ordered by the sum
# print(data.isnull().sum().sort_values(ascending=False))
