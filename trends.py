import pandas as pd
import numpy as np
import os

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from matplotlib.widgets import Button


trends_folder = "./Google_Trends_Data_Challenge_Datasets/trends"
prices_folder = "./Google_Trends_Data_Challenge_Datasets/prices"

# Function to load all CSV files in a folder into a dictionary of DataFrames
def load_csv_files(folder_path):
    data = {}
    for file in os.listdir(folder_path):
        if file.endswith('.csv'):
            file_path = os.path.join(folder_path, file)
            data[file] = pd.read_csv(file_path)
    return data

# Load data from both folders
trends_data = load_csv_files(trends_folder)
prices_data = load_csv_files(prices_folder)

# Print the first few rows of one file from each dataset
print("Trends Data Sample:")
print(trends_data[list(trends_data.keys())[0]].head())

print("\nPrices Data Sample:")
print(prices_data[list(prices_data.keys())[0]].head())

def eda(trends_data, prices_data):
    # Show the first few rows of trends and prices data
    print("Trends Data Structure:")
    first_crypto = list(trends_data.keys())[0]
    print(trends_data[first_crypto].head())

    # Extract cryptocurrency symbol from the trends file name
    crypto_symbol = first_crypto.split('.')[0].upper()

    # Find the matching price data file for the crypto_symbol
    price_file = f"{crypto_symbol}-USD.csv"

    # Check if the price file exists
    if price_file in prices_data:
        price_df = prices_data[price_file]
        print(price_df.head())
    else:
        print(f"Error: Price data for {crypto_symbol} not found.")

    # Check for missing data in trends and prices
    print("\nMissing Data in Trends:")
    for crypto, df in trends_data.items():
        print(f"{crypto}: {df.isnull().sum()}")
    
    print("\nMissing Data in Prices:")
    for crypto, df in prices_data.items():
        print(f"{crypto}: {df.isnull().sum()}")

eda(trends_data, prices_data)

def plot_all_crypto_graphs(trends_data, prices_data, crypto_list):
    """
    Plot search trends vs. price for multiple cryptocurrencies on a single page.
    
    Parameters:
        trends_data (dict): Dictionary containing trends dataframes.
        prices_data (dict): Dictionary containing prices dataframes.
        crypto_list (list): List of tuples with cryptocurrency names and symbols.
    """
    num_cryptos = len(crypto_list)
    num_cols = 2
    num_rows = (num_cryptos + num_cols - 1) // num_cols

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(16, 6 * num_rows))
    axes = axes.flatten()

    for i, (crypto_name, crypto_symbol) in enumerate(crypto_list):
        ax = axes[i]
        trend_file = f"{crypto_name}.csv"
        trend_df = trends_data.get(trend_file)
        price_file = f"{crypto_symbol}-USD.csv"
        price_df = prices_data.get(price_file)
        
        if trend_df is None or price_df is None:
            ax.set_title(f"Data not available for {crypto_name.capitalize()}")
            ax.axis('off')
            continue

        # Prepare the data
        trend_df['Week'] = pd.to_datetime(trend_df['Week'])
        trend_df.set_index('Week', inplace=True)
        price_df['Date'] = pd.to_datetime(price_df['Date'])
        merged_df = pd.merge(trend_df, price_df[['Date', 'Close']], left_index=True, right_on='Date', how='inner')

        if merged_df.empty:
            ax.set_title(f"No overlapping data for {crypto_name.capitalize()}")
            ax.axis('off')
            continue

        # Plot the data
        ax.plot(merged_df.index, merged_df[f"{crypto_name}: (Worldwide)"], label='Search Trends', color='b', alpha=0.7)
        ax.plot(merged_df['Date'], merged_df['Close'], label='Price (Close)', color='r', alpha=0.7)

        ax.set_title(f"{crypto_name.capitalize()} Search Trends vs Price")
        ax.set_xlabel('Date')
        ax.set_ylabel('Value')
        ax.legend(loc='upper left')
        ax.grid(True)
        ax.tick_params(axis='x', rotation=45)

    # Turn off unused subplots
    for j in range(len(crypto_list), len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()

crypto_list = [
    ("bitcoin", "BTC"),
    ("ethereum", "ETH"),
    ("cardano", "ADA"),
    ("chainlink", "LINK"),
    # ("fetch.ai", "FET"),
    # ("litecoin", "LTC"),
    # ("ocean protocol", "OCEAN"),
    # ("singularitynet", "AGIX"),
    # ("uniswap", "UNI"),
    # ("dogecoin", "DOGE"),
    # ("filecoin", "FIL"),
    # ("monero", "XMR"),
    # ("pancakeswap", "CAKE"),
    # ("XRP", "XRP"),
    # ("solana", "SOL"),
    # ("kucoin", "KCS"),
    # ("oasis network", "ROSE"),
    # ("polkadot", "DOT"),
    # ("tezos", "XTZ")
]


# Call the function
plot_all_crypto_graphs(trends_data, prices_data, crypto_list)


# Call the function
plot_all_crypto_graphs(trends_data, prices_data, crypto_list)


# Example dictionaries for trends and prices data
trends_data = {
    "bitcoin.csv": pd.read_csv("./Google_Trends_Data_Challenge_Datasets/trends/bitcoin.csv"),
    "ethereum.csv": pd.read_csv("./Google_Trends_Data_Challenge_Datasets/trends/ethereum.csv"),
    "cardano.csv": pd.read_csv("./Google_Trends_Data_Challenge_Datasets/trends/cardano.csv"),
    "chainlink.csv": pd.read_csv("./Google_Trends_Data_Challenge_Datasets/trends/chainlink.csv"),
    # "fetch.ai.csv": pd.read_csv("./Google_Trends_Data_Challenge_Datasets/trends/fetch.ai.csv"),
    # "litecoin.csv": pd.read_csv("./Google_Trends_Data_Challenge_Datasets/trends/litecoin.csv"),
    # "ocean protocol.csv": pd.read_csv("./Google_Trends_Data_Challenge_Datasets/trends/ocean protocol.csv"),
    # "singularitynet.csv": pd.read_csv("./Google_Trends_Data_Challenge_Datasets/trends/singularitynet.csv"),
    # "uniswap.csv": pd.read_csv("./Google_Trends_Data_Challenge_Datasets/trends/uniswap.csv"),
    # "dogecoin.csv": pd.read_csv("./Google_Trends_Data_Challenge_Datasets/trends/dogecoin.csv"),
    # "filecoin.csv": pd.read_csv("./Google_Trends_Data_Challenge_Datasets/trends/filecoin.csv"),
    # "monero.csv": pd.read_csv("./Google_Trends_Data_Challenge_Datasets/trends/monero.csv"),
    # "pancakeswap.csv": pd.read_csv("./Google_Trends_Data_Challenge_Datasets/trends/pancakeswap.csv"),
    # "XRP.csv": pd.read_csv("./Google_Trends_Data_Challenge_Datasets/trends/XRP.csv"),
    # "solana.csv": pd.read_csv("./Google_Trends_Data_Challenge_Datasets/trends/solana.csv"),
    # "kucoin.csv": pd.read_csv("./Google_Trends_Data_Challenge_Datasets/trends/kucoin.csv"),
    # "oasis network.csv": pd.read_csv("./Google_Trends_Data_Challenge_Datasets/trends/oasis network.csv"),
    # "polkadot.csv ": pd.read_csv("./Google_Trends_Data_Challenge_Datasets/trends/polkadot.csv"),
    # "tezos.csv": pd.read_csv("./Google_Trends_Data_Challenge_Datasets/trends/tezos.csv"),
}

prices_data = {
    "BTC-USD.csv": pd.read_csv("./Google_Trends_Data_Challenge_Datasets/prices/BTC-USD.csv"),
    "ETH-USD.csv": pd.read_csv("./Google_Trends_Data_Challenge_Datasets/prices/ETH-USD.csv"),
    "ADA-USD.csv": pd.read_csv("./Google_Trends_Data_Challenge_Datasets/prices/ADA-USD.csv"),
    "BNB-USD.csv": pd.read_csv("./Google_Trends_Data_Challenge_Datasets/prices/BNB-USD.csv"),
    # "CAKE-USD.csv": pd.read_csv("./Google_Trends_Data_Challenge_Datasets/prices/CAKE-USD.csv"),
    # "DOT-USD.csv": pd.read_csv("./Google_Trends_Data_Challenge_Datasets/prices/DOT-USD.csv"),
    # "FET-USD.csv": pd.read_csv("./Google_Trends_Data_Challenge_Datasets/prices/FET-USD.csv"),
    # "KCS-USD.csv": pd.read_csv("./Google_Trends_Data_Challenge_Datasets/prices/KCS-USD.csv"),
    # "LTC-USD.csv ": pd.read_csv("./Google_Trends_Data_Challenge_Datasets/prices/LTC-USD.csv"),
    # "ROSE-USD.csv": pd.read_csv("./Google_Trends_Data_Challenge_Datasets/prices/ROSE-USD.csv"),
    # "UNI-USD.csv": pd.read_csv("./Google_Trends_Data_Challenge_Datasets/prices/UNI-USD.csv"),
    # "XRP-USD.csv": pd.read_csv("./Google_Trends_Data_Challenge_Datasets/prices/XRP-USD.csv"),
    # "AGIX-USD.csv": pd.read_csv("./Google_Trends_Data_Challenge_Datasets/prices/AGIX-USD.csv"),
    # "FIL-USD.csv": pd.read_csv("./Google_Trends_Data_Challenge_Datasets/prices/FIL-USD.csv"),
    # "DOGE-USD.csv ": pd.read_csv("./Google_Trends_Data_Challenge_Datasets/prices/DOGE-USD.csv"),
    # "LINK-USD.csv": pd.read_csv("./Google_Trends_Data_Challenge_Datasets/prices/LINK-USD.csv"),
    # "OCEAN-USD.csv": pd.read_csv("./Google_Trends_Data_Challenge_Datasets/prices/OCEAN-USD.csv"),
    # "SOL-USD.csv": pd.read_csv("./Google_Trends_Data_Challenge_Datasets/prices/SOL-USD.csv"),
    # "XMR-USD.csv": pd.read_csv("./Google_Trends_Data_Challenge_Datasets/prices/XMR-USD.csv"),
    # "XTZ-USD.csv": pd.read_csv("./Google_Trends_Data_Challenge_Datasets/prices/XTZ-USD.csv"),
}

# def feature_engineering(trends_data, prices_data):
#     # List to hold features and target
#     X = []
#     y = []

#     # Iterate over each cryptocurrency's trend data
#     for crypto_name, trend_df in trends_data.items():
#         crypto_symbol = crypto_name.split('.')[0].upper()
#         price_file = f"{crypto_symbol}-USD.csv"
        
#         # Ensure price data exists for this crypto
#         if price_file not in prices_data:
#             print(f"Price data for {crypto_name} not found.")
#             continue

#         price_df = prices_data[price_file]

#         # Prepare the data (convert columns to datetime, set indices)
#         trend_df['Week'] = pd.to_datetime(trend_df['Week'])
#         trend_df.set_index('Week', inplace=True)
#         price_df['Date'] = pd.to_datetime(price_df['Date'])

#         # Merge trend data with price data on common date
#         merged_df = pd.merge(trend_df, price_df[['Date', 'Close']], left_index=True, right_on='Date', how='inner')

#         # Check if merged_df is empty
#         if merged_df.empty:
#             print(f"No overlapping data for {crypto_name}. Skipping.")
#             continue

#         # Feature engineering: Here you can create features such as lagged trends, moving averages, etc.
#         # Example: Using the search trends column and the price close column
#         features = merged_df[f"{crypto_name}: (Worldwide)"].values
#         target = merged_df['Close'].values

#         # Append the data for feature and target
#         X.append(features)
#         y.append(target)

#     # Convert lists to numpy arrays for model input
#     X = np.array(X)
#     y = np.array(y)

#     # Ensure that X is not empty before scaling
#     if X.shape[0] == 0 or X.shape[1] == 0:
#         raise ValueError("No valid data available for feature engineering. X is empty.")

#     # Flatten X if necessary and scale it
#     X = X.flatten().reshape(-1, 1)  # Flatten and reshape X to be a 2D array

#     # Apply scaling
#     from sklearn.preprocessing import StandardScaler
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)

#     return X_scaled, y


# X, y = feature_engineering(trends_data, prices_data)
