import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, time, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pprint import pprint

from Utils import *
from Indicators import *


class Dataset(object):
    """Object that holds the full dataset, applies data splitting and weekly rolling window function and produces on the fly visualisation."""
    def __init__(self, path, pair, params):
        
        # load and process full tick data
        self.tick_df = pd.read_csv(path)
        self.tick_df['Timestamp'] = pd.to_datetime(self.tick_df['Timestamp'])  # handle datatypes
        self.tick_df['Midprice'] = (self.tick_df['Bid'] + self.tick_df['Ask']) / 2  # add midprice
        self.tick_df['Spread'] = (self.tick_df['Ask'] - self.tick_df['Bid']) / self.tick_df['Ask']  # add spread
        self.tick_df.set_index('Timestamp', inplace=True)  # set Timestamp as index
        
        # set variables
        self.pair = pair
        self.params = params
        self.window_size = params['sampling']['window_size']
        self.shift = params['sampling']['shift']

        # split into weekly windows with train, val, test splits
        print("Sampling...")
        self.sampled_dfs = self.sample_windows()
        print(len(self.sampled_dfs))


        # profile data at each DC threshold
        self.transformed_dfs = {}
        for theta in params['sampling']['thetas']:
            self.transformed_dfs[theta] = {}
            print(f'Profiling {pair} at threshold {theta * 100}%')

            for idx, df_dict in enumerate(self.sampled_dfs):
                print(f"Profiling window {idx}")
                self.transformed_dfs[theta][f'Window_{idx}'] = {}
                for set_name, df in df_dict.items():
                    # DC indicators
                    profiled_df = self.profile_data(df, theta)
                    profiled_df = self.add_DC_indicators(profiled_df, theta)

                    # traditional indicators
                    tick_df = self.add_traditional_indicators(df.reset_index(drop=True))

                    # mergs dfs
                    profiled_df = pd.merge(profiled_df, tick_df, left_on='DCC Index', right_index=True)
                    self.transformed_dfs[theta][f'Window_{idx}'][set_name] = profiled_df

            print()

        # save to files
        self.save_data()

    def sample_windows(self):

        # get first and last timestamp of all ticks
        first_date = datetime.combine(self.tick_df.index[0].date(), time.min)
        last_date = datetime.combine(self.tick_df.index[-1].date(), time.max)

        # init loop
        timestamps = [first_date]
        next_timestamp = first_date

        # loop while the next timestamp is before the end timestamp
        while next_timestamp < last_date:
            next_timestamp += timedelta(weeks=self.shift)
            timestamps.append(next_timestamp)

        # generate samples based on if they are within the sliding 4 week ranges
        sampled_dfs = []
        for idx, timestamp in enumerate(timestamps[self.window_size:]):
            dfs = {}

            # select rows where index is between two dates for train, val and test sets
            week_dates = timestamps[idx: idx+self.window_size+1]

            dfs['train'] = self.tick_df.loc[(self.tick_df.index >= week_dates[0]) & (self.tick_df.index <= week_dates[2])]
            dfs['validation'] = self.tick_df.loc[(self.tick_df.index >= week_dates[2]) & (self.tick_df.index <= week_dates[3])]
            dfs['test'] = self.tick_df.loc[(self.tick_df.index >= week_dates[3]) & (self.tick_df.index <= week_dates[4])]

            sampled_dfs.append(dfs)

        return sampled_dfs


    def profile_data(self, df, theta):
        # direction: -1 is downturn, 1 is upturn
        starting_price = df['Midprice'][0]
        # direction, DC_start, DCC, OS_end, DC_start_index, DCC_index, OS_end
        trend_buffer = [1, starting_price, starting_price, starting_price, 0, 0, 0]
        trends = []

        # iterate over midprices
        for index, midprice in enumerate(df['Midprice'].values):

            # for upturn
            if trend_buffer[0] == 1:
                # threshold broken
                if pct_change(trend_buffer[3], midprice) < -trend_buffer[0] * theta:
                    # log old event
                    trends.append(Trend(*trend_buffer))
                    # setup new event
                    trend_buffer = [-1, trend_buffer[3], midprice, midprice, trend_buffer[6], index, index]
                # new extreme
                elif midprice > trend_buffer[3]:
                    trend_buffer[3], trend_buffer[6] = midprice, index

            # for downturn
            elif trend_buffer[0] == -1:
                # threshold broken
                if pct_change(trend_buffer[3], midprice) > -trend_buffer[0] * theta:
                    # log old event
                    trends.append(Trend(*trend_buffer))
                    # setup new event
                    trend_buffer = [1, trend_buffer[3], midprice, midprice, trend_buffer[6], index, index]
                # new extreme
                elif midprice < trend_buffer[3]:
                    trend_buffer[3], trend_buffer[6] = midprice, index
                    
        return pd.DataFrame([trend.data_dict for trend in trends])
        
    def add_traditional_indicators(self, df):
        func_dict = globals()
        info_dict = self.params['trad_dict']
        ticks = df['Midprice'].values
        for indicator, value in info_dict.items():
            # print(indicator)
            if isinstance(value, str):
                df[indicator] = func_dict[value]()
            elif isinstance(value, dict):
                for period_key, period_value in value.items():
                    if isinstance(period_value[0], int):
                        for period in period_value:
                            # print('\tperiod', period)
                            df[f'{indicator}_{period}'] = func_dict[list(value.keys())[0]](ticks, period)
                    elif isinstance(period_value[0], tuple):
                        for period_tuple in period_value:
                            # print('\tperiod', period_tuple)
                            df[f'{indicator}_{period_tuple}'] = func_dict[list(value.keys())[0]](ticks, *period_tuple)
        return df

    
    def add_DC_indicators(self, df, theta):

        func_dict = globals()
        trend_df = df
        info_dict = self.params['DC_dict']
        for indicator, value in info_dict.items():
            # print(indicator)
            if isinstance(value, str):
                args = (theta, None)
                df[indicator] = func_dict[value](trend_df, *args)
            elif isinstance(value, dict):
                # print(indicator)
                if isinstance(list(value.values())[0], list):
                    for period_list in value.values():
                        if isinstance(period_list[0], int):
                            for period in period_list:
                                # print('\tperiod', period)
                                df[f'{indicator}_{period}'] = func_dict[list(value.keys())[0]](trend_df, theta, period)

        return df

    def save_data(self):
        info_dict = {}
        for theta in self.transformed_dfs.keys():
            print(str(theta))
            info_dict[theta] = {}
            formatted_theta = '%.5f' % theta
            base_path = os.path.join('./TransformedData', formatted_theta, self.pair)
            for window in self.transformed_dfs[theta].keys():
                info_dict[theta][window] = {}
                path = os.path.join(base_path, window)
                os.makedirs(path, exist_ok=True)
                for set_name, df in self.transformed_dfs[theta][window].items():

                    # clean up df
                    df.drop(['End', 'Start Index', 'DCC Index', 'End Index', 'Pair', 'Midprice'], axis=1, inplace=True)
                    df = df.reset_index(drop=True)
                    final_nan_index = np.where(df.isna().any(axis=1))[0][-1]
                    df.drop(index=range(final_nan_index+1), inplace=True)

                    info_dict[theta][window][set_name] = len(df)
                    df.to_parquet(os.path.join(path, f'{set_name}.parquet.gzip'), compression='gzip')
            with open(os.path.join(base_path, 'info.json'), 'w') as f:
                json.dump(info_dict, f, indent=4)
        print("Data Saved")

    
    def feature_info(self, col_name):
        spread_dict = {
            'max': df[col_name].max(), 
            'min': df[col_name].min(), 
            'mean': df[col_name].mean(), 
            'median': df[col_name].median(), 
            'std dev': df[col_name].std(), 
            'skew': df[col_name].skew(), 
            'kurtosis': df[col_name].kurtosis()
        }
        
        pprint(spread_dict)
    
    def visualise(self, start_index=0, n_events=10):
        
        # set final tick index
        start_idx = int(self.trend_df['Start Index'].iloc[start_index])
        final_idx = int(self.trend_df['End Index'].iloc[start_index + n_events - 1])
        
        # get directional change data
        DC_coords = []
        OS_coords = []
        DCC_coords = []
        DCE_coords = []

        for index, row in self.trend_df.iloc[start_index: start_index + n_events].iterrows():
            DCC_line = [
                [self.tick_df['Timestamp'].iloc[int(row['Start Index'])], row['Start']], 
                [self.tick_df['Timestamp'].iloc[int(row['DCC Index'])], row['DCC']]
            ]
            OS_line = [
                [self.tick_df['Timestamp'].iloc[int(row['DCC Index'])], row['DCC']], 
                [self.tick_df['Timestamp'].iloc[int(row['End Index'])], row['End']]
            ]
            DCC_point = [self.tick_df['Timestamp'].iloc[int(row['DCC Index'])], row['DCC']]
            DCE_point = [self.tick_df['Timestamp'].iloc[int(row['End Index'])], row['End']]

            DC_coords.append(DCC_line)
            OS_coords.append(OS_line)
            DCC_coords.append(DCC_point)
            DCE_coords.append(DCE_point)
        
        # create a new figure with a specified size
        plt.figure(figsize=(20, 10))

        # format the x-axis tick labels
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        plt.gca().xaxis.set_major_locator(mdates.MinuteLocator())

        # plot tick data
        plt.plot(self.tick_df['Timestamp'].iloc[start_idx:final_idx], self.tick_df['Midprice'].iloc[start_idx:final_idx])

        # plot DC data
        for event in DC_coords:
            plt.plot([event[0][0], event[1][0]], [event[0][1], event[1][1]], 'b-')
        for event in OS_coords:
            plt.plot([event[0][0], event[1][0]], [event[0][1], event[1][1]], 'r-')

        # plot DC lables
        for DCC, label in zip(DCC_coords, ['DCC'] * len(DCC_coords)):
            plt.annotate(label, DCC, bbox=dict(facecolor='yellow', edgecolor='black', boxstyle='round'))
        for DCE, label in zip(DCE_coords, ['DCE']  * len(DCE_coords)):
            plt.annotate(label, DCE, bbox=dict(facecolor='green', edgecolor='black', boxstyle='round'))

        # add titles
        plt.title('DC Graph', fontsize=20)
        plt.xlabel('Timestamp', fontsize=10)
        plt.ylabel('Midprice', fontsize=10)
        
        # add information text
        info_text = f'Pair: AUDUSD\nThreshold: {self.theta}'
        plt.text(0.01, 0.95, info_text, transform=plt.gca().transAxes,
                 bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))

        # show the plot
        plt.show()