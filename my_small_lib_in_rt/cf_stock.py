import time
import os
import sys
import multiprocessing as mp
from numpy import exp, sqrt
import numpy as np
import pandas as pd
from loguru import logger

CUR_DIR = os.path.dirname(os.path.realpath(__file__))
if CUR_DIR not in sys.path:
    sys.path.insert(0, CUR_DIR)

import cf_util as cu
import cf_tradedate as ct

def eid_to_uid(eid):
    if eid[0] == '0' or eid[0] == '3':
        return eid + '-SZ-stock'
    elif eid[0] == '6':
        return eid + '-SH-stock'
    else:
        raise Exception(f'not supported eid: {eid}')


STOCK_BAR_MAP = dict()
def get_dolvol_and_vwap(uid, minute_strs, min_cum_dolvol=50000):
    cum_dolvol, cum_volume = 0, 0
    for minute_str in minute_strs:
        minute_obj = pd.to_datetime(minute_str)
        bar_path = '/data/hft/data_v1/dyndata/alpha_service/stock_bar_v5-1m/{}/{:02d}/{:02d}/{}.csv'.format(
            minute_obj.year, minute_obj.month, minute_obj.day, minute_obj.strftime('%H:%M:%S'))
        global STOCK_BAR_MAP
        if bar_path in STOCK_BAR_MAP:
            bar_df = STOCK_BAR_MAP[bar_path]
        else:
            if len(STOCK_BAR_MAP) > 1000:  # clear cash when there are too many entries
                print('STOCK_BAR_MAP full, clear')
                STOCK_BAR_MAP = {}
            print('reading from {}'.format(bar_path))
            bar_df = pd.read_csv(bar_path)
            STOCK_BAR_MAP[bar_path] = bar_df
        if len(bar_df.loc[bar_df['Uid'] == uid]) == 0:
            print('!!! can not find data for {} at {}'.format(uid, minute_obj))
            return cum_dolvol, np.nan
        bar_row = bar_df.loc[bar_df['Uid'] == uid].iloc[0]
        cum_dolvol += bar_row['Dolvol']
        cum_volume += bar_row['Volume']
    if cum_dolvol < min_cum_dolvol:
        return cum_dolvol, np.nan
    return cum_dolvol, cum_dolvol / cum_volume

def read_eod_df_of_date(tgt_date, eod_path=None):
    tgt_date = cu.dashed_date_str(tgt_date)
    if not eod_path:
        src_path = '/data/hft/data_v1/hftdata/WIND/static/stock/{}/{}/{}/eodprice.csv'.format(*tgt_date.split('-'))
    else:
        src_path = eod_path.format(*tgt_date.split('-'))

    df = pd.read_csv(src_path, dtype={'TradeDate': str})
    # exclude those records where `TradeDate` does not agree with `tgt_date`
    df['DashedTradeDate'] = df['TradeDate'].map(lambda d: cu.dashed_date_str(d))
    df = df.loc[df['DashedTradeDate'] == tgt_date]

    # df['AmtInWan'] = df['Amount'] * 1000 / 10000
    df['Amt'] = df['Amount'] * 1000
    df['AdjFactor'] = df['AdjClose'] / df['Close']
    df['AdjOpen'] = df['Open'] * df['AdjFactor']
    df['AdjHigh'] = df['High'] * df['AdjFactor']
    df['AdjLow'] = df['Low'] * df['AdjFactor']
    df['Date'] = tgt_date
    df = df[['Date', 'Uid', 'WindCode', 'Open', 'High', 'Low', 'Close', 'AdjOpen', 'AdjHigh', 'AdjLow', 'AdjClose',
             'Volume', 'Amt', 'AdjFactor']]
    return df

def read_eod_between(begin_date, end_date):
    tds = [cu.dashed_date_str(td) for td in ct.trading_dates_between(begin_date, end_date)]
    eod_dfs = []
    for td in tds:
        eod_df = read_eod_df_of_date(td)
        eod_dfs.append(eod_df)
    big_eod_df = pd.concat(eod_dfs, axis=0)
    return big_eod_df

def read_adjuster_df_of_date(tgt_date, adjuster_path=None):
    if not adjuster_path:
        src_path = '/data/hft/data_v1/hftdata/WIND/static/stock/{}/{}/{}/adjuster.csv'.format(*tgt_date.split('-'))
    else:
        src_path = adjuster_path.format(*tgt_date.split('-'))
    df = pd.read_csv(src_path)
    return df

def read_barra_df_of_date(tgt_date, barra_path=None):
    if not barra_path:
        src_path = '/data/hft/data_v1/hftdata/BARRA/static/stock/{}/{}/{}/CNE5S.RSK.csv'.format(*tgt_date.split('-'))
    else:
        src_path = barra_path.format(*tgt_date.split('-'))

    df = pd.read_csv(src_path, dtype={'Eid': str})
    df['Uid'] = df['Eid'].map(lambda x: eid_to_uid(x))
    return df

def read_stock_d1_bars(uid, begin_date, end_date):
    tds = ct.trading_dates_between(begin_date, end_date)
    tds = [cu.dashed_date_str(td) for td in tds]
    bar_dfs = []
    for td in tds:
        eod_df = read_eod_df_of_date(td)
        bar_df = eod_df.loc[eod_df['Uid'] == uid]
        bar_dfs.append(bar_df)
    big_bar_df = pd.concat(bar_dfs, axis=0)
    return big_bar_df

def read_stock_m1_bars(uid, tgt_date):
    begin_time, end_time = tgt_date + ' 09:30:00', tgt_date + ' 15:00:00'
    minutes = ct.trading_minutes_between(begin_time, end_time)
    bar_dfs = []
    for min_str in minutes:
        bar_path = '/data/hft/data_v1/dyndata/alpha_service/stock_bar_v5-1m/{}/{}/{}/{}.csv'.format(
            *tgt_date.split('-'), min_str.split(' ')[-1])
        bar_df = pd.read_csv(bar_path)
        bar_df = bar_df.loc[bar_df['Uid'] == uid]
        bar_dfs.append(bar_df)
    big_bar_df = pd.concat(bar_dfs, axis=0)
    # print('got minute bars:\n{}'.format(big_bar_df))
    return big_bar_df

def read_stock_m30_bar_of_date(tgt_date, uid=None):
    bar_path = '/data/hft/data_v1/dyndata/alpha_service/alpha_bar-30m/{}/{}/{}/level0_bar.csv'.\
        format(*tgt_date.split('-'))
    bar_df = pd.read_csv(bar_path, dtype={'SnapTime': str})
    if uid is not None:
        bar_df = bar_df.loc[bar_df['Uid'] == uid]
    return bar_df

def fill_adjustment_factor(df, tgt_date):
    eod_path = '/data/hft/data_v1/hftdata/WIND/static/stock/{}/{}/{}/eodprice.csv'.format(*tgt_date.split('-'))
    eod_df = pd.read_csv(eod_path)
    eod_df['AdjFactor'] = eod_df['AdjClose'] / eod_df['Close']
    df = df.merge(eod_df[['Uid', 'AdjFactor']], on=['Uid'], how='left')
    return df

def adjust_stock_price(df, uid_col, price_col, from_date, to_date):
    original_df = df
    df = df[[uid_col, price_col]].copy()
    assert len(df[uid_col].unique()) == len(df[uid_col]), 'adjust stock price, uid is not unique'
    from_eod_path = '/data/hft/data_v1/hftdata/WIND/static/stock/{}/{}/{}/eodprice.csv'.format(*from_date.split('-'))
    from_eod_df = pd.read_csv(from_eod_path, dtype={'TradeDate': str})
    from_eod_df = from_eod_df.loc[from_eod_df['TradeDate'] == cu.compact_date_str(from_date)]
    from_eod_df['FromAdjFactor'] = from_eod_df['AdjClose'] / from_eod_df['Close']
    # print(from_eod_df)
    to_eod_path = '/data/hft/data_v1/hftdata/WIND/static/stock/{}/{}/{}/eodprice.csv'.format(*to_date.split('-'))
    to_eod_df = pd.read_csv(to_eod_path, dtype={'TradeDate': str})
    to_eod_df = to_eod_df.loc[to_eod_df['TradeDate'] == cu.compact_date_str(to_date)]
    to_eod_df['ToAdjFactor'] = to_eod_df['AdjClose'] / to_eod_df['Close']
    # print(to_eod_df)
    df = df.merge(from_eod_df[['Uid', 'FromAdjFactor']], left_on=uid_col, right_on='Uid', how='left').\
        merge(to_eod_df[['Uid', 'ToAdjFactor']], left_on='Uid', right_on='Uid', how='left')

    new_price_col = price_col + '_Adj'
    df[new_price_col] = df[price_col] * df['FromAdjFactor'] / df['ToAdjFactor']
    original_df = original_df.merge(df[[uid_col, new_price_col]], on=uid_col)
    return original_df

WIND_CODE_TO_STOCK_NAME_DICT = None
def get_windcode_to_stockname_dict():
    global WIND_CODE_TO_STOCK_NAME_DICT
    if WIND_CODE_TO_STOCK_NAME_DICT is None:
        df = pd.read_csv(os.path.join(CUR_DIR, 'static/WIND_AShareDescription.csv'))
        WIND_CODE_TO_STOCK_NAME_DICT = df[['S_INFO_WINDCODE', 'S_INFO_NAME']].\
            set_index('S_INFO_WINDCODE')['S_INFO_NAME'].to_dict()
    return WIND_CODE_TO_STOCK_NAME_DICT

WIND_CODE_TO_COMP_NAME_DICT = None
def get_windcode_to_compname_dict():
    global WIND_CODE_TO_COMP_NAME_DICT
    if WIND_CODE_TO_COMP_NAME_DICT is None:
        df = pd.read_csv(os.path.join(CUR_DIR, 'static/WIND_AShareDescription.csv'))
        WIND_CODE_TO_COMP_NAME_DICT = df[['S_INFO_WINDCODE', 'S_INFO_COMPNAME']].\
            set_index('S_INFO_WINDCODE')['S_INFO_COMPNAME'].to_dict()
    return WIND_CODE_TO_COMP_NAME_DICT

def _single_read_stock_d1_factor_values(fname, td, factor_dir):
    if not factor_dir:
        src_path = '/data/current-rnd/data/cf/ws_stock/factor_values_d1/{}/{}/{}/{}.csv'.format(
            *td.split('-'), fname)
    else:
        src_path = '{}/{}/{}/{}/{}.csv'.format(factor_dir, *td.split('-'), fname)
    factor_df = pd.read_csv(src_path)
    return factor_df

def read_stock_d1_factor_values_between(fname, begin_date, end_date, factor_dir=None, parallel=False):
    # print('== reading factor {} data from {} to {}'.format(fname, begin_date, end_date))
    tds = [cu.dashed_date_str(td) for td in ct.trading_dates_between(begin_date, end_date)]
    if not parallel:
        factor_dfs = []
        for td in tds:
            if not factor_dir:
                src_path = '/data/rnd5/data/cf/ws_stock/factor_values_d1/{}/{}/{}/{}.csv'.format(
                    *td.split('-'), fname)
            else:
                src_path = '{}/{}/{}/{}/{}.csv'.format(factor_dir, *td.split('-'), fname)

            # print('read from {}'.format(src_path))
            factor_df = pd.read_csv(src_path)
            factor_dfs.append(factor_df)
    else:
        mp_params = [(fname, td, factor_dir) for td in tds]
        with mp.Pool(processes=20) as pool:
            factor_dfs = pool.starmap(_single_read_stock_d1_factor_values, mp_params)

    big_factor_df = pd.concat(factor_dfs, axis=0)
    return big_factor_df

def write_stock_d1_factor_values(dest_dir, tgt_date, factor_name, df, replace_inf,
                                 include_index=False, append_cols=False):
    if not dest_dir:
        raise Exception('write_stock_d1_factor_values, dest dir {} does not exist'.format(dest_dir))
    dir_with_date = os.path.join(dest_dir, '{}/{}/{}'.format(*tgt_date.split('-')))
    os.makedirs(dir_with_date, exist_ok=True)
    dest_path = os.path.join(dir_with_date, '{}.csv'.format(factor_name))

    cols = ['EndDate', 'ExecDate', 'Uid', factor_name]
    if append_cols:
        cols += ['DayClose', 'MktCapInHML']
    partial_df = df[cols]
    if replace_inf:
        partial_df = partial_df.replace(np.inf, np.nan).replace(-np.inf, np.nan)
        partial_df[factor_name] = partial_df[factor_name].fillna(partial_df[factor_name].mean())
    partial_df.to_csv(dest_path, index=include_index)
    logger.info('written {} values to {}'.format(factor_name, dest_path))

def d1_factor_value_exists(factor_name, tgt_date):
    dest_path = '/data/current-rnd/data/cf/ws_stock/factor_values_d1/{}/{}/{}/{}.csv'.format(
        *tgt_date.split('-'), factor_name)
    return os.path.exists(dest_path)



# def read_Y_d1_values_between(begin_date, end_date):
#     Y_dfs = []
#     for td in ct.trading_dates_between(begin_date, end_date):
#         td = cu.dashed_date_str(td)
#         src_path = '/data/current-rnd/data/cf/ws_stock/Y/Y_D1_{}.csv'.format(cu.compact_date_str(td))
#         Y_df = pd.read_csv(src_path)
#         Y_dfs.append(Y_df)
#     big_Y_df = pd.concat(Y_dfs, axis=0)
#     return big_Y_df

# def read_Y_d5_values_between(begin_date, end_date, source='znn'):
#     if source != 'znn':
#         raise Exception('not supported source')
#     years = ct.trading_years_between(begin_date, end_date)
#     year_dfs = []
#     for year in years:
#         year_df = pd.read_csv('/data/current-rnd/data/cf/ws_stock/data/Y/Y_D5_{}.csv'.format(year))
#         year_dfs.append(year_df)
#     big_year_df = pd.concat(year_dfs, axis=0)
#     big_year_df = big_year_df.loc[(big_year_df['ExecDate'] >= begin_date) & (big_year_df['ExecDate'] <= end_date)]
#     return big_year_df

def read_Y_0935_to_N0935_between(begin_date, end_date):
    Y_dfs = []
    for td in ct.trading_dates_between(begin_date, end_date):
        td = cu.dashed_date_str(td)
        src_path = '/data/current-rnd/data/cf/ws_stock/data/Y_0935_to_N0935/{}/{}/{}/Y.csv'.format(*td.split('-'))
        Y_df = pd.read_csv(src_path)
        Y_dfs.append(Y_df)
    big_Y_df = pd.concat(Y_dfs, axis=0)
    return big_Y_df

def read_Y_0935_to_T5_0935_between(begin_date, end_date):
    Y_dfs = []
    for td in ct.trading_dates_between(begin_date, end_date):
        td = cu.dashed_date_str(td)
        src_path = '/data/current-rnd/data/cf/ws_stock/data/Y_0935_to_T5_0935/{}/{}/{}/Y.csv'.format(*td.split('-'))
        Y_df = pd.read_csv(src_path)
        Y_dfs.append(Y_df)
    big_Y_df = pd.concat(Y_dfs, axis=0)
    return big_Y_df

def read_znn_0935_IndyRes_1800_5D_between(begin_date, end_date):
    years = ct.trading_years_between(begin_date, end_date)
    year_dfs = []
    for year in years:
        year_df = pd.read_csv('/data/current-rnd/data/cf/ws_stock/data/Y_D5_0935/Y_D5_{}.csv'.format(year))
        year_dfs.append(year_df)
    big_year_df = pd.concat(year_dfs, axis=0)
    big_year_df = big_year_df.loc[(big_year_df['ExecDate'] >= begin_date) & (big_year_df['ExecDate'] <= end_date)]
    return big_year_df


def read_yyh_y_1800_5D_0935_values_between(begin_date, end_date):
    tds = [cu.dashed_date_str(td) for td in ct.trading_dates_between(begin_date, end_date)]
    y_dfs = []
    for td in tds:
        src_path = '/data/current-rnd/data/cf/ws_stock/data/yyh_y_1800_5D/{}/{}/{}/Y.csv'.format(
            *td.split('-'))
        y_df = pd.read_csv(src_path)
        y_dfs.append(y_df)
    big_y_df = pd.concat(y_dfs, axis=0)
    return big_y_df

def get_formula_dict_by_bundle_name(bundle_name):
    from formula_cf_base import CF_BASE_DICT
    from formula_br import BR_DICT
    from formula_cf_lob import CF_LOB_DICT
    if bundle_name == 'all':
        ret_dict = dict()
        ret_dict.update(CF_BASE_DICT)
        ret_dict.update(BR_DICT)
        ret_dict.update(CF_LOB_DICT)
        return ret_dict
    elif bundle_name.lower() == 'cf_base':
        return CF_BASE_DICT
    elif bundle_name.lower() == 'br':
        return BR_DICT
    elif bundle_name.lower() == 'cf_lob':
        return CF_LOB_DICT
    else:
        raise Exception('unrecognized bundle name: {}'.format(bundle_name))






if __name__ == '__main__':
    # print('bond value: \n{}'.format(get_bond_value('127057.SZ', '2022-06-08')))
    # print('\n')
    # print('bond value: \n{}'.format(get_bond_value('113016.SH', '2022-06-08')))
    # print('\n')
    # print('bond value: \n{}'.format(get_bond_value('128015.SZ', '2021-09-02')))

    # print(get_corp_rate('2022-06-01', 'AAA', 1))
    pass
