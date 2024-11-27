# Deciphering_Crypto_Trends

# Cryptocurrency Trends and Prices Analysis
This project explores the relationship between cryptocurrency search trends (from Google Trends data) and their corresponding market prices. Using Python, this project loads data, conducts exploratory data analysis (EDA), and visualizes trends against prices.

# Project Structure
# Folders
  trends/: Contains CSV files with Google Trends data for cryptocurrencies.
  prices/: Contains CSV files with historical price data for cryptocurrencies.
  
# Files
  main.py: The main Python script containing the logic for data loading, EDA, and visualization.
  README.md: Documentation for the project.

# Prerequisites
Ensure you have the following installed:
  Python 3.7+

# Required Python libraries:
  pandas
  numpy
  matplotlib
  seaborn
  scikit-learn
  
# Install the dependencies using pip:
  pip install pandas numpy matplotlib seaborn scikit-learn
  
# Directory Structure
  Place the CSV files in the following folder structure:

  Google_Trends_Data_Challenge_Datasets/
  ├── trends/
  │   ├── bitcoin.csv
  │   ├── ethereum.csv
  │   ├── cardano.csv
  │   └── ...
  ├── prices/
  │   ├── BTC-USD.csv
  │   ├── ETH-USD.csv
  │   ├── ADA-USD.csv
  │   └── ...
  
# Features
# 1. Data Loading
  The project uses the load_csv_files function to load all CSV files from the trends and prices directories into dictionaries for easy manipulation.

# 2. Exploratory Data Analysis (EDA)
  The eda function performs:

  Inspection of trends and price data structures.
  Validation of missing data in both trends and price datasets.
  Data merging to check overlaps between trends and price data for consistency.
  
# 3. Visualization
  The plot_all_crypto_graphs function:

  Creates line plots showing Google Trends search popularity and market prices for specified cryptocurrencies.
  Supports multiple cryptocurrencies in a grid layout.
  Example Cryptocurrencies:
  Bitcoin (BTC)
  Ethereum (ETH)
  Cardano (ADA)
  Chainlink (LINK)
  Usage
  Run the Script
  
# To execute the main script, run:
  python main.py

# Visualization
  The script generates line plots comparing search trends with cryptocurrency prices. Example:

X-Axis: Date
Y-Axis: Search Trends and Price (Close)
Plots: Overlapping trends and prices for each cryptocurrency.
Customizing the Cryptocurrency List

# Update the crypto_list variable in the script to include more cryptocurrencies:
  crypto_list = [
      ("bitcoin", "BTC"),
      ("ethereum", "ETH"),
      ("cardano", "ADA"),
      ("chainlink", "LINK"),
  ]
  
# Example Output
# EDA Output
# Sample output showing the structure and missing data in the datasets:

# Trends Data Sample:
      Week  bitcoin: (Worldwide)
0  2021-01-03                   75
1  2021-01-10                   65
2  2021-01-17                   70

# Prices Data Sample:
        Date   Close
0 2021-01-03  33000.0
1 2021-01-10  36000.0
2 2021-01-17  34000.0

# Visualization Output
  Line plots for search trends and prices for selected cryptocurrencies.
  Clear trends showing correlations (or lack thereof) between search popularity and price fluctuations.
