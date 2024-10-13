import os
import sys
import numpy as np
import pandas as pd

CUR_DIR = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
if CUR_DIR not in sys.path:
    sys.path.insert(0, CUR_DIR)

import cf_util as cu


def cal_sharpe(df, col):
    return df[col].mean() / df[col].std() * np.sqrt(252)


def cal_max_dd(df, col):
    cummax_ser = df[col].cummax()
    dd_ser = (df[col] - cummax_ser) / cummax_ser
    return dd_ser.min()


def evaluate_ccbond_d1(df, col):
    df = cu.drop_nans(df, col)

    def cal_index_ret(g_df):
        g_df = g_df.copy()
        g_df['Weight'] = 1 / len(g_df)

        total_weight = g_df['Weight'].sum()
        ret = (g_df['Weight'] / total_weight * g_df[col]).sum()
        uid_count = len(g_df)
        return pd.Series(data=[ret, uid_count], index=[col, 'UidCount'])

    result_df = df.groupby('Date').apply(cal_index_ret).reset_index()

    # modify Return by Uid Count
    def mod_ret(row):
        # don't count in trading fees, let's see the theoretical performance of the strategy
        if row['UidCount'] >= 4:
            return row[col]
        else:
            return row[col] * row['UidCount'] / 4

    result_df['ModRet'] = result_df.apply(mod_ret, axis=1)

    result_df['UnitValue'] = result_df['ModRet'].cumsum() + 1
    # print(df.iloc[0]['INDNAME'], len(df))
    print(result_df)
    sharpe = cal_sharpe(result_df, 'ModRet')
    max_dd = cal_max_dd(result_df, 'UnitValue')
    print('sharpe: ', sharpe)
    print('max dd: ', max_dd)
    return result_df.iloc[-1]['UnitValue'], sharpe, max_dd, result_df


def evaluate_ccbond_m1(df, col):
    df['Date'] = df['DateTime_R'].map(lambda x: x.split(' ')[0])

    # def cal_index_ret(g_df):
    #     g_df = g_df.copy()
    #     g_df['Weight'] = 1 / len(g_df)
    #
    #     total_weight = g_df['Weight'].sum()
    #     ret = (g_df['Weight'] / total_weight * g_df[col]).sum()
    #     uid_count = len(g_df)
    #     return pd.Series(data=[ret, uid_count], index=[col, 'UidCount'])

    result_df = df.groupby('Date')[col].sum().reset_index()
    result_df['UnitValue'] = result_df[col].cumsum() + 1
    # print(result_df)

    sharpe = cal_sharpe(result_df, col)
    max_dd = cal_max_dd(result_df, 'UnitValue')
    # print('sharpe: ', sharpe)
    # print('max dd: ', max_dd)
    return result_df.iloc[-1]['UnitValue'], sharpe, max_dd, result_df
