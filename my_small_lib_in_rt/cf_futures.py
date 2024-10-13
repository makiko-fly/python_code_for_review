import time
import os
import sys
from numpy import exp, sqrt
import numpy as np
import pandas as pd
from loguru import logger

CUR_DIR = os.path.dirname(os.path.realpath(__file__))
if CUR_DIR not in sys.path:
    sys.path.insert(CUR_DIR)

import cf_util as cu
import cf_tradedate as ct



def get_leading_contract_uid_eid(product, tgt_date):
    hot_df = pd.read_csv('/data/hft/data_v1/hftdata/WIND/static/cta/{}/{}/{}/hot.csv.gz'.format(*tgt_date.split('-')))
    hot_df = hot_df.loc[hot_df['Product'] == product]
    row = hot_df.loc[(hot_df['StartDate'] <= tgt_date) & (hot_df['EndDate'] >= tgt_date)].iloc[0]
    return row['Uid'], row['Eid']

def get_secinfoam_row(eid, tgt_date):
    secinfoam_df = pd.read_csv('/data/hft/data_v1/hftdata/WIND/static/cta/'
                               '{}/{}/{}/secinfoam.csv'.format(*tgt_date.split('-')))
    secinfoam_row = secinfoam_df.loc[secinfoam_df['Eid'] == eid].iloc[0]
    return secinfoam_row

def read_commodity_future_level0_bars(eid, begin_date, end_date):
    tds = ct.trading_dates_between(begin_date, end_date)
    tds = [cu.dashed_date_str(td) for td in tds]
    bar_dfs = []
    for td in tds:
        bar_df = pd.read_csv('/data/hft/data_v1/dyndata/alpha_service/comdty_futures_bar_v2-1m'
                             '/{}/{}/{}/level0_bar.csv'.format(*td.split('-')))
        bar_dfs.append(bar_df)
    big_bar_df = pd.concat(bar_dfs)
    if eid != 'all':
        big_bar_df = big_bar_df.loc[big_bar_df['Eid'] == eid]
    return big_bar_df

def read_commodity_future_bars(eid, begin_date, end_date, freq):
    big_bar_df = read_commodity_future_level0_bars(eid, begin_date, end_date)

    big_bar_df.index = pd.to_datetime(big_bar_df['DateTime'])
    big_bar_df['_tod2'] = big_bar_df['_tod']

    resampled_df = big_bar_df.resample(rule=cu.to_pandas_freq(freq), closed='right', label='right').agg(
        {'Open': 'first', 'Close': 'last', 'High': 'max', 'Low': 'min', 'Vol': 'sum', '_tod': 'first',
         '_tod2': 'last'}).dropna().reset_index(drop=False)
    return resampled_df


if __name__ == '__main__':
    _, the_eid = get_leading_contract_uid_eid('I.DCE', '2019-12-31')
    print(read_commodity_future_bars(the_eid, '2019-12-11', '2019-12-31', 'D1'))

    pass
