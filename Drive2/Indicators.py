import numpy as np
from Utils import *

# all tick indicator functions get passed the array of tick values and respective variables

def moving_average(ticks, period):
    ma = np.convolve(ticks, np.ones(period) / period, mode='valid')
    padded_ma = np.concatenate((np.full(period-1, np.nan), ma))
    return padded_ma

def moving_sum(ticks, period):
    ms = np.convolve(ticks, np.ones(period), mode='valid')
    padded_ms = np.concatenate((np.full(period-1, np.nan), ms))
    return padded_ms

def rsi(ticks, period):
    delta = np.diff(ticks)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.zeros_like(ticks)
    avg_loss = np.zeros_like(ticks)
    avg_gain[period] = gain[:period].mean()
    avg_loss[period] = loss[:period].mean()
    for i in range(period + 1, len(ticks)):
        avg_gain[i] = ((period - 1) * avg_gain[i - 1] + gain[i - 1]) / period
        avg_loss[i] = ((period - 1) * avg_loss[i - 1] + loss[i - 1]) / period
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    rsi[:period] = np.nan
    return rsi

def macd(ticks, *args):
    fast_period, slow_period, signal_period = args[0], args[1], args[2]
    fast_ema = exponential_moving_average(ticks, fast_period, True)
    slow_ema = exponential_moving_average(ticks, slow_period, True)
    macd_line = fast_ema - slow_ema
    signal_line = exponential_moving_average(macd_line, signal_period, True)
    macd_histogram = macd_line - signal_line
    macd_histogram[:slow_period+signal_period] = np.nan
    return macd_histogram

def exponential_moving_average(ticks, period, macd=False):
    alpha = 2 / (period + 1)
    ema = np.zeros(len(ticks))
    ema[0] = ticks[0]
    for i in range(1, len(ticks)):
        ema[i] = alpha * ticks[i] + (1 - alpha) * ema[i - 1]
    if macd == False:
        ema[:period] = np.nan
    return ema


# all trend indicator functions get passed trend_df values and respective variables
# each output gets appended to trend_df

def TMV(df, *args):
    theta = args[0]
    starts, ends = df['Start'].values, df['End'].values
    total_changes = (ends - starts) / starts
    tmvs = np.abs(total_changes) / theta
    shifted_tmvs = np.empty_like(tmvs)
    shifted_tmvs[1:] = tmvs[:-1]
    shifted_tmvs[0] = np.nan
    return shifted_tmvs
    
def OSV(df, *args):
    theta = args[0]
    DCC = df['DCC'].values
    current_DCC = DCC
    current_change = np.diff(DCC) / DCC[:-1]
    normalised_change = current_change / theta
    normalised_change = np.insert(normalised_change, 0, np.nan)
    return normalised_change

def R_DC(df, *args):
    theta = args[0]
    tmvs = TMV(df, theta)
    prev_dcc = shift(df['DCC Index'].values, 1)
    current_dcc = df['DCC Index']
    trend_ticks = current_dcc - prev_dcc
    return np.divide((tmvs * theta), trend_ticks, out=np.zeros_like(trend_ticks), where=trend_ticks!=0)

def T_DC(df, *args):
    return shift(df['End Index'].values - df['Start Index'].values, 1)

def N_DC(df, *args):
    period = args[1]
    total_ticks = T_DC(df, ())
    ndcs = moving_average(total_ticks, period)
    return ndcs

def C_DC(df, *args):
    theta, period = args[0], args[1]
    tmvs = TMV(df, theta)

    # new code
    cdcs = moving_sum(tmvs, period)

    return cdcs


def A_T(df, *args):
    period = args[1]

    # new code
    diff = T_DC(df, ())
    direction = df['Direction'].values
    data = list(zip(direction, diff))

    period = period * 2
    rolling_sums = [np.nan] * (period - 1)
    
    # Iterate over the list using a sliding window approach
    for i in range(len(data) - period + 1):
        # Calculate the sum of the window
        pairs = data[i:i+period]
        up = [item[1] for item in pairs if item[0] == 1]
        down = [item[1] for item in pairs if item[0] == -1]
        sum_diff = np.sum(up) - np.sum(down)
        rolling_sums.append(sum_diff)

    return shift(rolling_sums, 1)

def average_OSV(df, *args):
    theta, period = args[0], args[1]
    osvs = OSV(df, theta)
    return moving_average(osvs, period)
    
def average_R_DC(df, *args):
    theta, period = args[0], args[1]
    rdcs = R_DC(df, theta)
    return moving_average(rdcs, period)

def average_T_DC(df, *args):
    period = args[1]
    tdcs = T_DC(df, ())
    return moving_average(tdcs, period)