import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from sklearn.linear_model import LinearRegression

def array_coef(x, y):
    x_values = np.array(x, dtype=float)
    y_values = np.array(y, dtype=float)
    x_values[np.isinf(x_values)] = np.nan
    y_values[np.isinf(y_values)] = np.nan
    nan_index = np.isnan(x_values) | np.isnan(y_values)
    x_values[nan_index] = np.nan
    y_values[nan_index] = np.nan
    delta_x = x_values - np.nanmean(x_values, axis=0)
    delta_y = y_values - np.nanmean(y_values, axis=0)
    multi = np.nanmean(delta_x * delta_y, axis=0) / (np.nanstd(delta_x, axis=0) * np.nanstd(delta_y, axis=0))
    multi[np.isinf(multi)] = np.nan
    return pd.Series(multi, index=x.columns)



def rolling_corr(df_x, df_y, window=None):
    """"""
    assert df_x.shape[0] == df_y.shape[0], 'dims must be same'

    corr = pd.DataFrame(np.nan, index=df_x.index, columns=df_x.columns)

    if window == None or window <= 0:
        window = df_x.shape[0]
    if window <= df_x.shape[0] and window > 1:
        for idx, index in enumerate(df_x.index):
            if idx >= window - 1:
                corr.loc[index] = array_coef(df_x.iloc[idx - window + 1:idx + 1],
                                             df_y.iloc[idx - window + 1:idx + 1]).values
    return corr


def array_column_wise_regress(df_x, df_y):
    x_values = np.array(df_x, dtype=float)
    y_values = np.array(df_y, dtype=float)
    x_values[np.isinf(x_values)] = np.nan
    y_values[np.isinf(y_values)] = np.nan

    mean_x = np.nanmean(x_values, axis=0)
    mean_y = np.nanmean(y_values, axis=0)
    delta_x = x_values - mean_x
    delta_y = y_values - mean_y
    beta = np.nanmean(delta_x * delta_y, axis=0) / np.nanstd(delta_x, axis=0) ** 2
    beta[np.isinf(beta)] = np.nan
    const = mean_y - beta * mean_x
    resid = np.subtract(y_values - np.multiply(x_values, beta), const)

    beta = pd.Series(beta, index=df_x.columns)
    resid = pd.DataFrame(resid, index=df_x.index, columns=df_x.columns)
    return beta, resid


def rolling_process(factor_df: pd.DataFrame, ptype: str, window: int = None, min_periods: int = None):
    factor_df_new = factor_df.dropna(how='all', axis=0)
    factor_df_new = factor_df_new.replace(np.inf, np.nan)
    factor_df_new = factor_df_new.replace(-np.inf, np.nan)

    if not window:
        raise ValueError('Rolling window should be specified.')
    if not min_periods:
        min_periods = window

    if ptype == 'raw':
        factor_df_new = factor_df_new
    elif ptype == 'mean':
        factor_df_new = factor_df_new.rolling(window=window, min_periods=min_periods).apply(np.nanmean, raw=True)
    elif ptype == 'std':
        factor_df_new = factor_df_new.rolling(window=window, min_periods=min_periods).apply(np.nanstd, raw=True)
    elif ptype == "meandivstd":
        temp1 = factor_df_new.rolling(window=window, min_periods=min_periods).apply(np.nanmean, raw=True)
        temp2 = factor_df_new.rolling(window=window, min_periods=min_periods).apply(np.nanstd, raw=True)
        factor_df_new = temp1 / temp2
        factor_df_new = factor_df_new.replace({np.inf: np.nan, -np.inf: np.nan})
    elif ptype == "diff":
        factor_df_new = factor_df_new - factor_df_new.shift(periods=window)
    elif ptype == 'diffdivstd':
        factor_df_new_std = factor_df_new.rolling(window).std()
        factor_df_new_std[factor_df_new == 0] = np.nan
        factor_df_new = (factor_df_new - factor_df_new.shift(periods=window)) / factor_df_new_std
    elif ptype == 'skew':
        factor_df_new = factor_df_new.rolling(window=window, min_periods=min_periods).apply(skew, raw=True)  # 默认是无偏估计量
    elif ptype == 'kurt':
        factor_df_new = factor_df_new.rolling(window=window, min_periods=min_periods).apply(kurtosis,
                                                                                            raw=True)  # 默认是无偏估计量
    elif ptype == 'sr':
        factor_df_new = factor_df_new.rolling(window=window, min_periods=min_periods).apply(
            lambda x: np.nanmean(x) / np.nanstd(x), raw=True)  # sharpe
    elif ptype == 'max':
        factor_df_new = factor_df_new.rolling(window=window, min_periods=min_periods).apply(np.nanmax, raw=True)
    elif ptype == 'min':
        factor_df_new = factor_df_new.rolling(window=window, min_periods=min_periods).apply(np.nanmin, raw=True)
    elif ptype == 'dif':
        factor_df_new = factor_df_new.rolling(window=window, min_periods=min_periods).apply(
            lambda x: np.nanmax(x) - np.nanmin(x), raw=True)  # 最大值-最小值
    elif ptype == 'pct':
        temp_df = np.divide(factor_df_new, factor_df_new.shift(window)) - 1  # window天变化率
        factor_df_new = pd.DataFrame(temp_df, index=factor_df_new.index.to_list(),
                                     columns=factor_df_new.columns.to_list())
    elif ptype == 'rank':
        factor_df_new = factor_df_new.rank(axis=1, pct=True)  # 截面排序
    elif ptype == "bias":
        temp_df = factor_df_new.rolling(window=window, min_periods=min_periods).apply(np.nanmean, raw=True)
        factor_df_new = factor_df_new - temp_df
    elif ptype == 'abs':
        factor_df_new = factor_df_new.abs()
    elif ptype == 'regbeta':
        regbeta_func = lambda col: get_regression_params(range(window), col)[1]
        factor_df_new = factor_df_new.rolling(window=window, min_periods=min_periods).apply(regbeta_func,
                                                                                            raw=True)  # 回归beta
    elif ptype == 'regrmse':
        def regrmse_func(col):
            x, y = np.array(range(window)), col
            alpha, beta = get_regression_params(x, y)
            y_hat = alpha + x * beta
            rmse = np.sqrt(np.nanmean(np.power(y - y_hat, 2)))
            return rmse

        factor_df_new = factor_df_new.rolling(window=window, min_periods=min_periods).apply(regrmse_func,
                                                                                            raw=True)  # 回归RMSE
    elif ptype == 'regres':
        def regres_func(col):
            x, y = np.array(range(window)), col
            alpha, beta = get_regression_params(x, y)
            res = y[-1] - (alpha + beta * x[-1])
            return res

        factor_df_new = factor_df_new.rolling(window=window, min_periods=min_periods).apply(regres_func,
                                                                                            raw=True)  # 回归残差
    else:
        raise ValueError('Unknown process type {}.'.format(ptype))
    return factor_df_new


def get_regression_params(x, y):
    x, y = np.array(x), np.array(y)
    x_ = x[~(np.isnan(x) | np.isnan(y))]
    y_ = y[~(np.isnan(x) | np.isnan(y))]
    if len(x_) / len(x) < 0.5 or len(x_) < 3:
        return np.nan, np.nan
    beta = np.cov(y_, x_, bias=True)[0, 1] / np.var(x_)
    alpha = np.mean(y_) - beta * np.mean(x_)
    return alpha, beta


def standardize(df, ismdf=False, n=3):
    # df is a dataframe with columns of stock names and rows are tradingDays
    if ismdf:
        col = df.columns[0]
        df = df[col].unstack()
    df = df.replace(np.inf, np.nan)
    df = df.replace(-np.inf, np.nan)
    m = df.mean(axis=1)
    s = df.std(axis=1, ddof=0)
    df1 = df.subtract(m, axis=0).divide(s, axis=0)
    df1[df1 > n] = n
    df1[df1 < -n] = -n
    dfnew = df1.multiply(s, axis=0).add(m, axis=0)
    dfs = dfnew.subtract(dfnew.mean(axis=1), axis=0).divide(dfnew.std(axis=1, ddof=0), axis=0)
    if ismdf:
        dfs = pd.DataFrame(dfs.stack(), columns=[col])
    return dfs


#TODO 暂时只是根据金工组的临时方案
def risk_neutral(mdf, mrisk):
    # mdf is a multiIndex, and mrisk is also a multiIndex
    data = mdf.join(mrisk)
    tradingDays = data.index.get_level_values(level=0).unique()
    alpname = mdf.columns
    risknames = mrisk.columns

    malplist = []
    for td in tradingDays:
        dailyData1 = data.loc[[td]]
        # get res
        dailyData = dailyData1[~dailyData1.isin([np.nan, np.inf, -np.inf]).any(1)]
        # lm = linear_model.LinearRegression()
        lm = LinearRegression()  ################
        y = dailyData[alpname].values

        risknamesrep = []
        risknamesrep[:] = risknames

        x = dailyData[risknamesrep].values
        if 'Industry' in risknames:
            risknamesrep.remove('Industry')
            ind = pd.get_dummies(dailyData['Industry'])
            ind.columns = ['ind' + str(int(i)) for i in ind.columns]
            if 'ind0' in ind.columns:
                ind = ind.drop('ind0', 1)
            indsum = ind.sum()
            smallnumcols = indsum[indsum < 3].index
            ind = ind.drop(smallnumcols, 1)

            x = pd.concat([dailyData[risknamesrep], ind], axis=1).values

        lm.fit(x, y)
        res = y - np.dot(x, lm.coef_.T)
        npstd = np.nanstd(res)
        if npstd == 0:
            npstd = 1
        res = (res - np.nanmean(res)) / npstd
        alp_df = pd.DataFrame(res, index=dailyData.index, columns=alpname)

        # set alp_df to multiIndex
        malplist = malplist + [alp_df]
    malp = pd.concat(malplist)
    return malp


def data_filter(data_df, filter_df, method='day'):
    """
    将停牌/涨跌停过滤掉
    """
    ans_df = data_df.copy()
    if method == 'day':
        threshold = 0.5
        ans_df[filter_df.reindex(index=ans_df.index, columns=ans_df.columns).values > threshold] = np.nan
        ans_df[np.isnan(filter_df)] = np.nan
    elif method == 'minute':
        ans_df[abs(filter_df.reindex(index=ans_df.index, columns=ans_df.columns)).values > np.exp(-10)] = np.nan
        ans_df[np.isnan(filter_df)] = np.nan
    else:
        raise Exception("method only suport for day or minute!")
    return ans_df

