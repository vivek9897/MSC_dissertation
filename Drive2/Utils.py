import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

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