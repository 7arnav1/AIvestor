import yfinance as yf
import pandas as pd

# Define the stocks you want to analyze
stocks = ['AAPL', 'TSLA', 'MSFT']  # Apple, Tesla, Microsoft

# Define the date range for historical data
start_date = '2015-01-01'
end_date = '2024-01-01'

# Fetch the historical data for each stock
def fetch_stock_data(tickers, start, end):
    stock_data = {}
    for ticker in tickers:
        print(f"Fetching data for {ticker}...")
        stock_data[ticker] = yf.download(ticker, start=start, end=end)
    return stock_data

# Fetch data
stock_data = fetch_stock_data(stocks, start_date, end_date)

# Combine data into a single DataFrame
def combine_close_prices(stock_data):
    close_prices = pd.DataFrame()
    for ticker, data in stock_data.items():
        close_prices[ticker] = data['Close']
    return close_prices

# Get close prices for all stocks
close_prices = combine_close_prices(stock_data)
print(close_prices.head())

# Save the data to a CSV file
close_prices.to_csv('stock_data.csv', index=True)