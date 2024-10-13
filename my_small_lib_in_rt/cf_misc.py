import datetime as dt
import pathlib

import pandas as pd
import cf_util as cu
import cf_tradedate as ct
import cf_quant as cq


def to_windcode(uid):
    if uid.endswith('ccbond'):
        return uid.split('-')[0] + '.' + uid.split('-')[1]




def read_daily_close_df(begin_date, end_date):
    tds = ct.trading_dates_between(begin_date, end_date)
    tds = [cu.dashed_date_str(td) for td in tds]
    eod_dfs = []
    for td in tds:
        td = cu.dashed_date_str(td)
        eod_path = '/data/hft/data_v1/hftdata/WIND/static/stock/{}/{}/{}/eodprice.csv'.format(*td.split('-'))
        eod_df = pd.read_csv(eod_path, dtype={'TradeDate': str})
        eod_df = eod_df.loc[eod_df['TradeDate'] == cu.compact_date_str(td)]  # exclude halted stocks
        eod_dfs.append(eod_df)
    big_eod_df = pd.concat(eod_dfs, axis=0)
    daily_close_df = big_eod_df.set_index(['TradeDate', 'Uid'])[['AdjClose']].unstack(level=1).droplevel(0, axis=1)
    return daily_close_df

