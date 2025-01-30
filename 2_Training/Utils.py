import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import MinMaxScaler

# set seeds
seed = 42
random.seed(seed)
np.random.seed(seed)

# calculate profit
def calculate_profit(position_size, trade_direction, entry_price, exit_price):
  
    # fixed transaction cost
    t_cost = 0.0001 * position_size
  
    price_change = (exit_price - entry_price) / entry_price

    if trade_direction == 1:
        profit = price_change * position_size
    elif trade_direction == -1:
        profit = -price_change * position_size 
    else:
        return 0
    return profit - t_cost
    # return profit

# get datasets
def get_data(pair, window, theta):

    train_df = pd.read_parquet(f'/Users/vivekkumar/Documents/MSC_dissertation_project/TransformedData/{theta}/{pair}/Window_{window}/train.parquet.gzip')
    val_df = pd.read_parquet(f'/Users/vivekkumar/Documents/MSC_dissertation_project/TransformedData/{theta}/{pair}/Window_{window}/validation.parquet.gzip')
    test_df = pd.read_parquet(f'/Users/vivekkumar/Documents/MSC_dissertation_project/TransformedData/{theta}/{pair}/Window_{window}/test.parquet.gzip')

    # get ask prices
    train_asks = train_df['Ask'].values
    val_asks = val_df['Ask'].values
    test_asks = test_df['Ask'].values

    # get bid prices
    train_bids = train_df['Bid'].values
    val_bids = val_df['Bid'].values
    test_bids = test_df['Bid'].values

    # train_df, val_df, test_df = normalize_dataframes(train_df, val_df, test_df)
    train_df, val_df, test_df = manual_normalise(train_df, val_df, test_df)

    # temp remove to match old approach
    train_df = train_df.drop(['Direction', 'Ask', 'Bid', 'Spread'], axis=1)
    val_df = val_df.drop(['Direction', 'Ask', 'Bid', 'Spread'], axis=1)
    test_df = test_df.drop(['Direction', 'Ask', 'Bid', 'Spread'], axis=1)

    if 'JPY' in pair:
        train_df[['Start', 'DCC']] = train_df[['Start', 'DCC']] / 100
        val_df[['Start', 'DCC']] = val_df[['Start', 'DCC']] / 100
        test_df[['Start', 'DCC']] = test_df[['Start', 'DCC']] / 100

    # print(train_df.head)
    # print(train_df.columns.to_list())

    train_data = train_df.values
    val_data = val_df.values
    test_data = test_df.values

    data_dict = {
        'train_data': train_data, 
        'train_asks': train_asks, 
        'train_bids': train_bids,
        'val_data': val_data, 
        'val_asks': val_asks, 
        'val_bids': val_bids,
        'test_data': test_data, 
        'test_asks': test_asks,
        'test_bids': test_bids
    }

    print("------- Data -------")
    print(f"Train Shape: {train_data.shape}")
    print(f"Validation Shape: {val_data.shape}")
    print(f"Test Shape: {test_data.shape}")

    return data_dict

# percentage change
def pct_change(old_value, new_value):
    change = new_value - old_value
    percentage_change = (change / old_value)
    return percentage_change

# shift by n, new start values are replace by np.nan and end values are discarded
def shift(array, shift):
    return np.concatenate(([np.nan] * shift, array[:-shift]))

# rolling window generator, left over events are discarded
def rolling_window(df, window_size, shift):
    for i in range(0, len(df) - window_size + 1, shift):
        yield df.iloc[i:i+window_size]
        
# take train, validation and test sets and normalise based on training data
def normalize_dataframes(train_df, val_df, test_df):
    # create a MinMaxScaler object
    scaler = MinMaxScaler()

    # fit the scaler on the train DataFrame
    scaler.fit(train_df)

    # normalize each DataFrame using the fitted scaler
    train_normalized = pd.DataFrame(scaler.transform(train_df), columns=train_df.columns)
    val_normalized = pd.DataFrame(scaler.transform(val_df), columns=val_df.columns)
    test_normalized = pd.DataFrame(scaler.transform(test_df), columns=test_df.columns)

    return train_normalized, val_normalized, test_normalized

def manual_normalise(train_df, val_df, test_df):
    
    DC_start_end = train_df[['Start', 'DCC']]
    train_transformed = (train_df - train_df.min()) / (train_df.max() - train_df.min())
    train_transformed[['Start', 'DCC']] = DC_start_end

    DC_start_end = val_df[['Start', 'DCC']]
    val_transformed = (val_df - train_df.min()) / (train_df.max() - train_df.min())
    val_transformed[['Start', 'DCC']] = DC_start_end

    DC_start_end = test_df[['Start', 'DCC']]
    test_transformed = (test_df - train_df.min()) / (train_df.max() - train_df.min())
    test_transformed[['Start', 'DCC']] = DC_start_end

    return train_transformed, val_transformed, test_transformed


# trend class
class Trend(object):
    def __init__(self, direction, DC_start, DCC, OS_end, DC_start_index, DCC_index, OS_end_index):
        self.direction, self.DC_start, self.DCC, self.OS_end = direction, DC_start, DCC, OS_end
        self.DC_start_index, self.DCC_index, self.OS_end_index = DC_start_index, DCC_index, OS_end_index
        
        self.data_dict = {
                'Direction': self.direction, 
                'Start': round(self.DC_start, 6),
                'DCC': round(self.DCC, 6),
                'End': round(self.OS_end, 6),
                'Start Index': round(self.DC_start_index, 6),
                'DCC Index': round(self.DCC_index, 6),
                'End Index': round(self.OS_end_index, 6),
            }
        
    def __str__(self):
        return str(self.data_dict)