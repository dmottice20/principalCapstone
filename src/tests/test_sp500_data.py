import pandas as pd
import matplotlib.pyplot as plt
import os

# Set the paths
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(current_dir))
spx_path = os.path.join(root_dir, 'src', 'managers', 'data', 'SPX.csv')
fred_path = os.path.join(root_dir, 'processed_fred_data.csv')

# Read FRED data
fred_data = pd.read_csv(fred_path)
fred_data['Date'] = pd.to_datetime(fred_data['Date'])
fred_data.set_index('Date', inplace=True)

# Read SPX data and align with FRED data start date
spx_data = pd.read_csv(spx_path)
spx_data['Date'] = pd.to_datetime(spx_data['Date'])
spx_data.set_index('Date', inplace=True)
spx_data = spx_data[spx_data.index >= fred_data.index.min()]

# Plot the data
plt.figure(figsize=(12, 8))
plt.plot(spx_data.index, spx_data['Close'], label='SPX Close')
# plt.plot(spx_data.index, spx_data['Adj Close'], label='SPX Adj Close')
plt.plot(fred_data.index, fred_data['SP500'], label='FRED SP500')

plt.title('Comparison of SPX and FRED SP500 Data')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Print some statistics for comparison
print("Date range:", spx_data.index.min(), "to", spx_data.index.max())
print("SPX Close vs FRED SP500 correlation:", spx_data['Close'].corr(fred_data['SP500']))
print("SPX Adj Close vs FRED SP500 correlation:", spx_data['Adj Close'].corr(fred_data['SP500']))
