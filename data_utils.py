# data_utils.py

import yfinance as yf
import numpy as np

def download_data(ticker, start_date):
    """Download stock data using yfinance."""
    try:
        data = yf.download(ticker, start=start_date, interval='1d')
        if data.empty:
            return None
        return data
    except Exception as e:
        print(f"Error downloading data for {ticker}: {e}")
        return None

def prepare_data(ticker, start_date):
    """Prepare and clean the data for backtesting."""
    data = download_data(ticker, start_date)
    if data is not None:
        # Ensure the data has the necessary columns and rename if necessary
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        data.columns = [col.lower() for col in data.columns]  # Convert columns to lowercase for consistency
        if all(col in data.columns for col in ['open', 'high', 'low', 'close', 'volume']):
            # Calculate returns
            data['returns'] = np.log(data['close'] / data['close'].shift(1)).fillna(0)
            return data
        else:
            print(f"Data is missing required columns. Found columns: {data.columns}")
            return None
    else:
        print("Failed to download data.")
        return None
