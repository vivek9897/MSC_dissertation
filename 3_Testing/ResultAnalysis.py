import sys
import os
import json
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

pairs = ['AUDJPY', 'AUDUSD', 'CADJPY', 'EURCHF', 'EURGBP', 'EURJPY', 'EURUSD', 'GBPUSD', 'NZDUSD', 'USDCAD', 'USDCHF', 'USDJPY']
# thresholds = [str(round(0.00011 + i * 0.00002, 5)) for i in range(9)]
thresholds = [0.00015]

# save metric aggs
def analyse(metric):
    metric = metric[~np.isinf(metric)]
    # analyse data
    stats = {
        'Mean': metric.mean(), 'Median': metric.median(), 'Variance': metric.var(), 'Range': metric.max() - metric.min(), 
        'Max': metric.max(), 'Min': metric.min(), 'Q1': metric.quantile(0.25), 'Q3': metric.quantile(0.75), 
        'Skewness': metric.skew(), 'Kurtosis': metric.kurtosis(),
    }
    return stats

for theta in thresholds:
    print(theta)

    result_data = {}

    # load data
    for pair in pairs:
        # set path
        path = f'./Results/{theta}/Val/{pair}.json'
        try:
            with open(path, 'r') as f:
                result_data[pair] = json.load(f)

        except:
            print(f'{pair} not logged')

    # reformat into metrics
    metric_dict = {}

    # dict init
    for metrics in result_data.values():
        for metric in list(metrics.keys()):
            metric_dict[metric] = {}

    # dict fill
    for pair, metrics in result_data.items():
        for metric, results in metrics.items():
            metric_dict[metric][pair] = results
            
    # convert to dataframe
    for metric in metric_dict.keys():
        metric_dict[metric] = pd.DataFrame(metric_dict[metric])

    # aggregate results
    agg_dict = {}
    for metric in metric_dict.keys():
        if metric == 'All Trades':
            break
        else:
            agg_dict[metric] = pd.DataFrame(analyse(metric_dict[metric]), index=pairs)

    # save results
    result_dir = f'./ResultAnalysis/'
    os.makedirs(result_dir, exist_ok=True)
    writer = pd.ExcelWriter(os.path.join(result_dir, f'ResultStatistics_{theta}.xlsx'), engine='xlsxwriter')
    for sheet_name, sheet_df in metric_dict.items():
        sheet_df.to_excel(writer, sheet_name=sheet_name, index=True)
    for sheet_name, sheet_df in agg_dict.items():
        sheet_df.to_excel(writer, sheet_name=f'{sheet_name}_agg', index=True)
    writer.save()


################################################################################
# load data
result_data = {}
for theta in thresholds:
    result_data[theta] = {}
    for pair in pairs:
        # set path
        path = f'./Results/{theta}/Val/{pair}.json'
        try:
            with open(path, 'r') as f:
                result_data[theta][pair] = json.load(f)
        except:
            print(f'{pair} not logged')

metrics = list(result_data['0.00015']['AUDUSD']['All Trades'].keys())

# get result by metric
metric_results = {}
for metric in metrics:
    metric_results[metric] = {}
    for theta in thresholds:
        metric_results[metric][theta] = {}
        for pair in pairs:
            metric_results[metric][theta][pair] = result_data[theta][pair]['All Trades'][metric]

    metric_results[metric] = pd.DataFrame(metric_results[metric])

# save metric results
writer = pd.ExcelWriter(os.path.join(result_dir, 'ThetaResultStatistics.xlsx'), engine='xlsxwriter')
for sheet_name, sheet_df in metric_results.items():
    sheet_df.to_excel(writer, sheet_name=sheet_name, index=True)
writer.save()
