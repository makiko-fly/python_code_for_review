import sys
import cf_util as cu
import numpy as np
import pandas as pd


def clean_and_check_X_ser(tgt_date, X_ser, raise_err=True):
    # replace inf values with NaN
    X_ser = X_ser.replace(np.inf, np.nan)
    zero_ser = X_ser[X_ser == 0]
    nan_ser = X_ser[X_ser.isna()]
    if len(zero_ser) / len(X_ser) > 0.2:
        if raise_err:
            raise Exception('too many zero series for {}'.format(tgt_date))
    nan_threshold = 0.2
    if tgt_date < '2016-01-01':
        nan_threshold = 0.3
    if len(nan_ser) / len(X_ser) > nan_threshold:
        if raise_err:
            raise Exception('too many nan series for {}, ser:\n{}'.format(tgt_date, nan_ser))
    return X_ser


def drop_outliers(xy_df, x_name, threshold):
    mean, std = xy_df[x_name].mean(), xy_df[x_name].std()
    assert not pd.isna(std) and std > 0, 'invalid std value: {}, series: \n{}'.format(std, xy_df[x_name])
    ret_df = xy_df.loc[(xy_df[x_name] - mean).abs() / std < threshold]
    return ret_df


def linear_rank(ser, min_records=300):
    if len(ser[~pd.isna(ser)]) < min_records:
        return pd.Series([np.nan]*len(ser), index=ser.index)
    else:
        return ser.rank() / (ser.count()+1) - 0.5


def rank_x_df(x_df, factor_name, by='DateTime'):
    converted_dfs = []
    for name, group in x_df.groupby(by):
        group = group.copy()
        group[factor_name] = linear_rank(group[factor_name], min_records=100)
        print('ranked single df:\n{}'.format(group))
        converted_dfs.append(group)
    x_df = pd.concat(converted_dfs, axis=0)
    return x_df

# Contrary to some suggestions like panel ranking, here we are looking at all X values as a whole.
# It's a complicated thought process, the basic argument is that the more data we look at, the more accurate
# its std reflect factor's real distribution. This might be forward looking but we are not back testing, it shows
# factor's real historical predicting power, in today's perspective.
def normalize_x(x_ser, cap=2):
    x_ser = clip_extreme_x(x_ser)
    mean, std = x_ser.mean(), x_ser.std()
    x_ser = (x_ser - mean) / std
    x_ser.clip(lower=-cap, upper=cap)
    print('after normalizing, x_ser:\n{}'.format(x_ser))
    return x_ser

# why a separate method to calculate a single day's X?
# Because the logic is not the same as calculating a period's data
def normalize_single_day_x(x_ser, cap=2):
    x_ser = clip_extreme_x(x_ser)
    mean, std = x_ser.mean(), x_ser.std()
    x_ser = (x_ser - mean) / std
    x_ser = x_ser.clip(lower=-cap, upper=cap)
    return x_ser


def clip_extreme_x(x_ser, method='median', max_distance=5):
    assert len(x_ser) > 300, 'clip extreme x, too few samples, len(x_ser): {}'.format(len(x_ser))
    assert len(x_ser.unique()) / len(x_ser) > 0.01, 'should not happen, x_ser: \n{}'.format(x_ser)
    distance_threshold = max_distance
    x_ser = x_ser.copy()
    print('== before clipping x_ser: \n{}'.format(x_ser.describe()))
    # print('== x_ser:\n{}'.format(x_ser))
    median_value = x_ser.median()
    median_distance = (x_ser - median_value).abs().median()
    q_02_val, q_98_val = list(x_ser.quantile([0.02, 0.98]))
    q_02_val_dis = abs(q_02_val - median_value) / median_distance
    q_98_val_dis = abs(q_98_val - median_value) / median_distance
    print('== median_value: {}, median_distance:{}, min_value: {}, q_02_val: {}, q_02_val_dis: {}, '
          'q_98_val: {}, q_98_val_dis: {}, max_value: {}'.
          format(median_value, median_distance, x_ser.min(), q_02_val, q_02_val_dis,
                 q_98_val, q_98_val_dis, x_ser.max()))
    if q_02_val_dis > distance_threshold:
        lower = q_02_val
    else:
        lower = median_value - distance_threshold * median_distance
    if q_98_val_dis > distance_threshold:
        upper = q_98_val
    else:
        upper = median_value + distance_threshold * median_distance
    lower_clipped = len(x_ser.loc[x_ser < lower])
    upper_clipped = len(x_ser.loc[x_ser > upper])
    print('== lower_clipped: {}, ratio: {:.3f}'.format(lower_clipped, lower_clipped / len(x_ser)))
    print('== upper_clipped: {}, ratio: {:.3f}'.format(upper_clipped, upper_clipped / len(x_ser)))
    x_ser.loc[x_ser < lower] = lower
    x_ser.loc[x_ser > upper] = upper
    print('== after clipping x_ser: \n{}'.format(x_ser.describe()))
    return x_ser


# def normalize_x(df, x_name, date_col='Date'):  # df should have Date, Uid and x_name as columns
#     # filt_df = df.loc[~df[x_name].isna()]
#     # filt_df = drop_outliers(filt_df, x_name, 3)
#     filt_df = df
#     ranked_ser = filt_df.groupby(date_col).apply(lambda day_df: linear_rank(day_df[x_name]))
#     assert len(filt_df) == len(ranked_ser), 'should not happen'
#     filt_df[x_name] = list(ranked_ser)
#     return filt_df


# sample df:
#             Uid       Ret  IndCode       Weight
# 0     000001.SZ -0.004025   6133.0  5797.503801
# 1     000002.SZ -0.010574   6118.0  4779.526999
# ...         ...       ...      ...          ...
# 3674  603998.SH -0.017123   6115.0   496.480271
# 3675  603999.SH -0.018553   6131.0   552.521493
def calc_ind_res_ret(df):
    df = cu.drop_nans(df)
    ind_ret_ser = df.groupby('IndCode').apply(lambda x: (x['Weight'] * x['Ret']).sum() / x['Weight'].sum())
    ind_ret_df = ind_ret_ser.to_frame().reset_index(drop=False)
    ind_ret_df.columns = ['IndCode', 'IndRet']
    df = df.merge(ind_ret_df, on='IndCode', how='left')

    df = df.set_index('Uid')
    return df['Ret'] - df['IndRet']


def ewma(df, hl, min_periods=1, require_recent=1):
    if hl == 0:
        return df
    ret_df = df.ewm(halflife=hl, axis=0, min_periods=min_periods).mean()  # axis=0 here
    # # set to NaN where df is NaN
    # df_copy = df.copy()
    # df_copy = df_copy.applymap(lambda x: np.nan if pd.isna(x) else 1)
    # ret_df = ret_df * df_copy / df_copy
    mask_df = df.isna().rolling(require_recent).sum() > 0
    ret_df[mask_df] = np.nan
    return ret_df


def sma(df, n, min_periods=1, require_recent=1):
    ret_df = df.rolling(n, min_periods=min_periods).mean()  # axis=0 here
    # # set to NaN where df is NaN
    # df_copy = df.copy()
    # df_copy = df_copy.applymap(lambda x: np.nan if pd.isna(x) else 1)
    # ret_df = ret_df * df_copy / df_copy
    mask_df = df.isna().rolling(require_recent).sum() > 0
    ret_df[mask_df] = np.nan
    return ret_df


def high_minus_low(df, window, min_periods=1):
    max_df = df.rolling(window, min_periods=min_periods).max()
    min_df = df.rolling(window, min_periods=min_periods).min()
    ret_df = (max_df - min_df) / min_df
    return ret_df


def _cal_RS(ret_ser):
    mean_inc = np.sum(ret_ser) / len(ret_ser)
    deviations = ret_ser - mean_inc
    Z = np.cumsum(deviations)
    R = max(Z) - min(Z)
    S = np.std(ret_ser, ddof=1)
    if R == 0 or S == 0 or pd.isna(R) or pd.isna(S):
        return 0
    return R / S


def cal_hurst_exponent(price_ser, min_window=8, max_window=None):
    # refer to https://github.com/Mottl/hurst
    assert len(price_ser) > 25, 'price series too short'
    if price_ser.isna().values.any():
        return [np.nan] * 3
    ret_ser = price_ser.pct_change(fill_method=None).iloc[1:]
    series = ret_ser
    max_window = max_window or len(series)-1

    window_sizes = list(map(
        lambda x: int(10**x),
        np.arange(np.log10(min_window), np.log10(max_window), 0.25)))
    window_sizes.append(len(series))

    RS = []
    for w in window_sizes:
        rs_list = []
        for start in range(0, len(series), w):
            if (start+w) > len(series):
                break
            rs = _cal_RS(series[start:start+w])
            if rs != 0:
                rs_list.append(rs)
        RS.append(np.mean(rs_list))

    A = np.vstack([np.log10(window_sizes), np.ones(len(RS))]).T
    H, c = np.linalg.lstsq(A, np.log10(RS), rcond=-1)[0]

    c = 10**c
    return H, c, [window_sizes, RS]


# def get_rsi(ser, halflife):
#     assert len(ser) >= halflife*8, 'price series length should be at least 8*halflife, price_ser:\n{}'.format(ser)
#     delta = ser.diff()
#     up = delta.clip(lower=0)
#     down = -1 * delta.clip(upper=0)
#     # print('got up:\n', up)
#     # print('got down:\n', down)
#     ema_up = up.ewm(halflife=halflife, adjust=False, min_periods=halflife * 4).mean().iloc[-1]
#     ema_down = down.ewm(halflife=halflife, adjust=False, min_periods=halflife * 4).mean().iloc[-1]
#     # print('got ema_down: \n{}'.format(ema_down))
#     if ema_down == 0:
#         return 100
#     else:
#         rs = ema_up / ema_down
#         return 100 - (100 / (1 + rs))

def cal_simple_rsi(ser):
    delta = ser.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    up_mean = up.mean()
    down_mean = down.mean()
    rsi = up_mean / (up_mean + down_mean)

    # amp = ser.max() / ser.min() - 1
    amp = 1
    mod_rsi = rsi * amp
    return mod_rsi


def cal_ewma_rsi(ser, alpha):
    delta = ser.diff()
    # print(delta)
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    up_mean = up.ewm(alpha=alpha, adjust=True).mean().iloc[-1]  # use adjust=True, why? refer to Pandas docs
    down_mean = down.ewm(alpha=alpha, adjust=True).mean().iloc[-1]
    rsi = up_mean / (up_mean + down_mean)

    # amp = ser.max() / ser.min() - 1
    amp = 1
    mod_rsi = rsi * amp
    # print('up: \n', up)
    # print('up ewm: \n', up.ewm(alpha=alpha, adjust=False).mean())
    # # print(down.ewm(alpha=alpha, adjust=False).mean())
    # print('up_mean: {}, down_mean: {}, rsi: {}'.format(up_mean, down_mean, mod_rsi))
    # if ser.index[-1] == '2021-12-30 10:41:00':
    #     sys.exit(0)
    return mod_rsi

def collect_stats(val_df, col):
    all_count = len(val_df)
    valid_count = len(cu.drop_nans(val_df, col))
    valid_ratio = valid_count / all_count
    mean = val_df[col].mean()
    std = val_df[col].std()
    min_ = val_df[col].min()
    max_ = val_df[col].max()
    skew = val_df[col].skew()
    kurt = val_df[col].kurtosis()
    q_list = [0.01, 0.025, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.975, 0.99]
    quantile_ser = val_df[col].quantile(q_list)
    q1 = quantile_ser[q_list[0]]
    q2_5 = quantile_ser[q_list[1]]
    q10 = quantile_ser[q_list[2]]
    q20 = quantile_ser[q_list[3]]
    q30 = quantile_ser[q_list[4]]
    q40 = quantile_ser[q_list[5]]
    q50 = quantile_ser[q_list[6]]
    q60 = quantile_ser[q_list[7]]
    q70 = quantile_ser[q_list[8]]
    q80 = quantile_ser[q_list[9]]
    q90 = quantile_ser[q_list[10]]
    q97_5 = quantile_ser[q_list[11]]
    q99 = quantile_ser[q_list[12]]
    result_ser = pd.Series(data=(all_count, valid_count, valid_ratio, mean, std, min_, max_, skew, kurt, q1, q2_5, q10, q20,
                                 q30, q40, q50, q60, q70, q80, q90, q97_5, q99),
                           index=['AllCount', 'ValidCount', 'ValidRatio', 'Mean', 'Std', 'Min', 'Max',
                                  'Skew', 'Kurtosis', 'Q_0.01', 'Q_0.025', 'Q_0.1',
                                  'Q_0.2', 'Q_0.3', 'Q_0.4', 'Q_0.5', 'Q_0.6', 'Q_0.7', 'Q_0.8', 'Q_0.9',
                                  'Q_0.975', 'Q_0.99'])
    result_df = result_ser.to_frame().T.copy()
    result_df['ColName'] = col
    result_df = cu.move_cols_to_front(result_df, ['ColName'])
    return result_df

