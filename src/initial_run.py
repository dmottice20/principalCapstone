from managers.data_manager import FREDDataManager
from managers.data_processing import DataProcessor
import json
import pandas as pd
from managers.cross_validation import TimeSeriesCrossValidator


USE_SAVED_CSV = True
USE_SAVED_PROCESSED_CSV = True

if not USE_SAVED_CSV:
    api_key = ''
    start_date = '1998-01-01'
    series_ids = [value for value in json.load(open('src/managers/data/fred-tickers.json')).values() if value is not None]
    fred_manager = FREDDataManager(api_key)
    all_data = fred_manager.get_all_data(series_ids, start_date, save_csv=True, csv_filename='fred_data_output.csv')
else:
    # Read the CSV file, setting 'Date' as the index and parsing dates
    all_data = pd.read_csv('fred_data_output.csv', index_col='Date', parse_dates=True)

print(all_data.describe())
print("Shape of the data: ", all_data.shape)

if not USE_SAVED_PROCESSED_CSV:
    processor = DataProcessor(all_data)
    target_columns = ['BAMLHYH0A0HYM2TRIV', 'BAMLCC0A0CMTRIV', "DGS10", "DGS2", "DGS30", "DGS5", "DGS7", "DGS1", "DGS20"]

    processed_data = processor.process_data(target_columns, save_csv=True, csv_filename='processed_fred_data.csv')
    print(processed_data.describe())
    print("Shape of the processed data: ", processed_data.shape)
else:
    processed_data = pd.read_csv('processed_fred_data.csv', index_col='Date', parse_dates=True)
    print(processed_data.describe())
    print("Shape of the processed data: ", processed_data.shape)

# Perform cross-validation
cv = TimeSeriesCrossValidator(processed_data)

# For rolling window
rolling_splits = cv.perform_cross_validation(method='rolling', initial_train_size=10, step_size=10, test_size=5)

# For sliding window
sliding_splits = cv.perform_cross_validation(method='sliding', window_size=25, step_size=10, test_size=5)

# Print information about the splits
print(f"Number of rolling window splits: {len(rolling_splits)}")
print(f"Number of sliding window splits: {len(sliding_splits)}")

# You can now use these splits for your model training and evaluation
for i, (train, test) in enumerate(rolling_splits):
    print(f"Rolling split {i+1}:")
    print(f"Train shape: {train.shape}, Test shape: {test.shape}")
    print(f"Train date range: {train.index.min()} to {train.index.max()}")
    print(f"Test date range: {test.index.min()} to {test.index.max()}")
    print()

# Similarly, you can iterate through sliding_splits
for i, (train, test) in enumerate(sliding_splits):
    print(f"Sliding split {i+1}:")
    print(f"Train shape: {train.shape}, Test shape: {test.shape}")
    print(f"Train date range: {train.index.min()} to {train.index.max()}")
    print(f"Test date range: {test.index.min()} to {test.index.max()}")
    print()
