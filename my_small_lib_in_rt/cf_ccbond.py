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
APP_DIR = os.path.join(CUR_DIR, '..')

import cf_util as cu
import cf_tradedate as ct
import cf_option as co
import cf_quant as cq


BOND_CF_DF = None
def get_bond_value(windcode, date, ret_detail=False):
    global BOND_CF_DF
    if BOND_CF_DF is None:
        BOND_CF_DF = pd.read_csv(os.path.join(APP_DIR, 'cf_data/WIND_CBondCF.csv'), dtype={'B_INFO_PAYMENTDATE': str})

    cf_df = BOND_CF_DF.loc[BOND_CF_DF['S_INFO_WINDCODE'] == windcode].sort_values(by='B_INFO_PAYMENTDATE')
    # print('cash flow df:\n{}'.format(cf_df))
    end_date = cf_df.iloc[-1]['B_INFO_PAYMENTDATE']
    term = ct.num_of_calendar_days(date, end_date) / 365
    rating = get_bond_rating(windcode, date)
    corp_rate = get_corp_rate(date, rating, term)
    value, payments = 0, []
    for _, cf_row in cf_df.iterrows():
        if cf_row['B_INFO_PAYMENTDATE'] >= cu.compact_date_str(date):
            t = ct.num_of_calendar_days(date, cf_row['B_INFO_PAYMENTDATE']) / 365
            # xuzw multiplied interest payment by 0.8 to account for tax
            if cf_row['B_INFO_PAYMENTSUM'] > 100:
                interest_payment = (cf_row['B_INFO_PAYMENTSUM'] - 100) * 0.8
                total_payment = 100 + interest_payment
            else:
                total_payment = cf_row['B_INFO_PAYMENTSUM'] * 0.8
            discounted_val = cf_row['B_INFO_PAYMENTSUM'] / (1 + corp_rate) ** t
            # print('row: \n{}'.format(cf_row.to_frame().T))
            # print('discounted value: {}'.format(discounted_val))
            value += discounted_val
            payments.append('{:.02f}'.format(cf_row['B_INFO_PAYMENTSUM']))
    if ret_detail:
        if corp_rate == 0:
            return np.nan, rating, corp_rate, term, ','.join(payments)
        else:
            return value, rating, corp_rate, term, ','.join(payments)
    else:
        if corp_rate == 0:
            return np.nan
        else:
            return value


def get_bond_value_xuzw(windcode, date, ret_detail=False):
    global BOND_CF_DF
    if BOND_CF_DF is None:
        BOND_CF_DF = pd.read_csv(os.path.join(APP_DIR, 'cf_data/WIND_CBondCF.csv'), dtype={'B_INFO_PAYMENTDATE': str})

    cf_df = BOND_CF_DF.loc[BOND_CF_DF['S_INFO_WINDCODE'] == windcode].sort_values(by='B_INFO_PAYMENTDATE')
    # # print('cash flow df:\n{}'.format(cf_df))
    # end_date = cf_df.iloc[-1]['B_INFO_PAYMENTDATE']
    # term = ct.num_of_calendar_days(date, end_date) / 365
    rating = get_bond_rating(windcode, date)
    # value, payments = 0, []
    value, discounted_values, payments, payments_after_tax, terms, corp_rates = 0, [], [], [], [], []
    for _, cf_row in cf_df.iterrows():
        payment_date = cf_row['B_INFO_PAYMENTDATE']
        if payment_date > cu.compact_date_str(date):  # use > instead of >=, 03-02 payment date, will pay on 03-01
            term = ct.num_of_calendar_days(date, payment_date) / 365
            corp_rate = get_corp_rate(date, rating, term)
            # xuzw multiplied interest payment by 0.8 to account for tax
            if cf_row['B_INFO_PAYMENTSUM'] > 100:
                interest_payment = (cf_row['B_INFO_PAYMENTSUM'] - 100) * 0.8
                payment_after_tax = 100 + interest_payment
            else:
                payment_after_tax = cf_row['B_INFO_PAYMENTSUM'] * 0.8
            discounted_val = payment_after_tax / (1 + corp_rate) ** term
            # print('row: \n{}'.format(cf_row.to_frame().T))
            # print('discounted value: {}'.format(discounted_val))
            value += discounted_val
            discounted_values.append('{:.02f}'.format(discounted_val))
            payments.append('{:.02f}'.format(cf_row['B_INFO_PAYMENTSUM']))
            payments_after_tax.append('{:.02f}'.format(payment_after_tax))
            terms.append('{:.02f}'.format(term))
            corp_rates.append('{:.03f}'.format(corp_rate))
    if ret_detail:
        return value, rating, ','.join(corp_rates), ','.join(terms), ','.join(payments), \
               ','.join(payments_after_tax), ','.join(discounted_values)
    else:
        return value


BOND_RATING_DF = None
def get_bond_rating(windcode, date):
    global BOND_RATING_DF
    if BOND_RATING_DF is None:
        BOND_RATING_DF = pd.read_csv(os.path.join(APP_DIR, 'cf_data/WIND_CBondRating.csv'), dtype={'ANN_DT': str})
    rating_df = BOND_RATING_DF.loc[BOND_RATING_DF['S_INFO_WINDCODE'] == windcode].sort_values(by='ANN_DT')
    return rating_df.loc[rating_df['ANN_DT'] <= cu.compact_date_str(date)].iloc[-1]['B_INFO_CREDITRATING']


def get_corp_rate(date, rating, term):
    yield_curve_path = '/data/new_share/hftdataman/yieldCurve/{}/{}/{}/{}.csv'.format(*date.split('-'), rating)
    if not os.path.exists(yield_curve_path) and rating in ['BBB-', 'B+', 'B-']:
        yield_curve_path = '/data/new_share/hftdataman/yieldCurve/{}/{}/{}/{}.csv'.format(*date.split('-'), 'BBB')
    df = pd.read_csv(yield_curve_path, dtype={'year': float})
    df = df.dropna(subset=['value']).sort_values(by='year')
    ret_val = df.loc[df['year'] > term].iloc[0]['value'] / 100
    print('got corp rate {} for {}, date: {}, term: {}'.format(ret_val, rating, date, term))
    return ret_val


def get_ccbond_close_at_minute(uid, minute_obj, use_vwap=False):
    minute_obj = pd.to_datetime(minute_obj)
    bar_path = '/data/hft/data_v1/dyndata/alpha_service/ccbond_bar_v5-1m/{}/{:02d}/{:02d}/{}.csv'.format(
        minute_obj.year, minute_obj.month, minute_obj.day, minute_obj.strftime('%H:%M:%S'))
    print('reading from {}'.format(bar_path))
    bar_df = pd.read_csv(bar_path)
    if len(bar_df.loc[bar_df['Uid'] == uid]) == 0:
        print('!!! can not find data for {} at {}'.format(uid, minute_obj))
        return np.nan
    bar_row = bar_df.loc[bar_df['Uid'] == uid].iloc[0]

    if bar_row['Dolvol'] == 0:  # if there is no trade, consider this invalid, probably halted trading
        return np.nan
    if use_vwap:
        return bar_row['Dolvol'] / bar_row['Volume']
    else:
        return bar_row['Close']


def get_minute_bars_for_single_ccbond(uid, minute_strs):
    bar_rows = []
    for minute_str in minute_strs:
        minute_obj = pd.to_datetime(minute_str)
        bar_path = '/data/hft/data_v1/dyndata/alpha_service/ccbond_bar_v5-1m/{}/{:02d}/{:02d}/{}.csv'.format(
            minute_obj.year, minute_obj.month, minute_obj.day, minute_obj.strftime('%H:%M:%S'))
        global CCBOND_BAR_MAP
        if bar_path in CCBOND_BAR_MAP:
            bar_df = CCBOND_BAR_MAP[bar_path]
        else:
            if len(CCBOND_BAR_MAP) > 1000:  # clear cash when there are too many entries
                print('CCBOND_BAR_MAP full, clear')
                CCBOND_BAR_MAP = {}
            print('reading from {}'.format(bar_path))
            bar_df = pd.read_csv(bar_path)
            CCBOND_BAR_MAP[bar_path] = bar_df
        if len(bar_df.loc[bar_df['Uid'] == uid]) == 0:
            print('!!! can not find data for {} at {}, return empty dataframe'.format(uid, minute_obj))
            return pd.DataFrame()
        bar_row = bar_df.loc[bar_df['Uid'] == uid]
        bar_rows.append(bar_row)
    df = pd.concat(bar_rows, axis=0)
    return df


def get_hurst_exponent(uid, minute_strs):
    df = get_minute_bars_for_single_ccbond(uid, minute_strs)
    # df['VWAP'] = df['Dolvol'] / df['Volume']
    ret_val, _, _ = cq.cal_hurst_exponent(df['Close'])
    print(ret_val)
    return ret_val

CCBOND_BAR_MAP = dict()
def get_dolvol_and_vwap(uid, minute_strs, min_cum_dolvol=50000):
    cum_dolvol, cum_volume = 0, 0
    for minute_str in minute_strs:
        minute_obj = pd.to_datetime(minute_str)
        bar_path = '/data/hft/data_v1/dyndata/alpha_service/ccbond_bar_v5-1m/{}/{:02d}/{:02d}/{}.csv'.format(
            minute_obj.year, minute_obj.month, minute_obj.day, minute_obj.strftime('%H:%M:%S'))
        global CCBOND_BAR_MAP
        if bar_path in CCBOND_BAR_MAP:
            bar_df = CCBOND_BAR_MAP[bar_path]
        else:
            if len(CCBOND_BAR_MAP) > 1000:  # clear cash when there are too many entries
                print('CCBOND_BAR_MAP full, clear')
                CCBOND_BAR_MAP = {}
            print('reading from {}'.format(bar_path))
            bar_df = pd.read_csv(bar_path)
            CCBOND_BAR_MAP[bar_path] = bar_df
        if len(bar_df.loc[bar_df['Uid'] == uid]) == 0:
            print('!!! can not find data for {} at {}'.format(uid, minute_obj))
            return cum_dolvol, np.nan
        bar_row = bar_df.loc[bar_df['Uid'] == uid].iloc[0]
        cum_dolvol += bar_row['Dolvol']
        cum_volume += bar_row['Volume']
    if cum_dolvol < min_cum_dolvol:
        return cum_dolvol, np.nan
    return cum_dolvol, cum_dolvol / cum_volume

def get_twap(uid, minute_strs):
    big_bar_df = read_ccbond_m1_bars(uid, minute_strs)
    big_bar_df['VWAP'] = big_bar_df['Dolvol'] / big_bar_df['Volume']
    return big_bar_df['VWAP'].mean()


CCBOND_TO_STOCK_MAP = None
def get_related_stock_uid(ccbond_uid):
    global CCBOND_TO_STOCK_MAP
    if CCBOND_TO_STOCK_MAP is None:
        df = pd.read_csv('/data/current-rnd/data/cf/ws_ccbond/data/ccbond_to_stock_map.csv')
        CCBOND_TO_STOCK_MAP = df.set_index('Uid')['StockUid'].to_dict()
    return CCBOND_TO_STOCK_MAP[ccbond_uid]

def get_volatility(uid, minute_strs):
    record_dfs = []
    for minute_str in minute_strs:
        minute_obj = pd.to_datetime(minute_str)
        bar_path = '/data/hft/data_v1/dyndata/alpha_service/ccbond_bar_v5-1m/{}/{:02d}/{:02d}/{}.csv'.format(
            minute_obj.year, minute_obj.month, minute_obj.day, minute_obj.strftime('%H:%M:%S'))
        global STOCK_BAR_MAP
        if bar_path in CCBOND_BAR_MAP:
            bar_df = CCBOND_BAR_MAP[bar_path]
        else:
            if len(CCBOND_BAR_MAP) > 1000:  # clear cash when there are too many entries
                print('CCBOND_BAR_MAP full, clear')
                CCBOND_BAR_MAP = {}
            print('reading from {}'.format(bar_path))
            bar_df = pd.read_csv(bar_path)
            CCBOND_BAR_MAP[bar_path] = bar_df
        uid_record_df = bar_df.loc[bar_df['Uid'] == uid][['RecordTime', 'Volume', 'Dolvol']].copy()
        if len(uid_record_df) == 0:
            print('!!! can not find data for {} at {}'.format(uid, minute_obj))
            continue
        assert len(uid_record_df) == 1, 'duplicate record for {}'.format(uid)
        uid_record_df['VWAP'] = uid_record_df['Dolvol'] / uid_record_df['Volume']
        record_dfs.append(uid_record_df)
    big_record_df = pd.concat(record_dfs, axis=0)
    if len(cu.drop_nans(big_record_df)) < 10:
        return np.nan
    return co.get_hist_sigma(big_record_df['VWAP'], 'simple')

def read_ccbond_m1_bars(uid, minute_strs):
    bar_dfs = []
    for minute_str in minute_strs:
        minute_obj = pd.to_datetime(minute_str)
        bar_path = '/data/hft/data_v1/dyndata/alpha_service/ccbond_bar_v5-1m/{}/{:02d}/{:02d}/{}.csv'.format(
            minute_obj.year, minute_obj.month, minute_obj.day, minute_obj.strftime('%H:%M:%S'))
        global CCBOND_BAR_MAP
        if bar_path in CCBOND_BAR_MAP:
            bar_df = CCBOND_BAR_MAP[bar_path]
        else:
            if len(CCBOND_BAR_MAP) > 1000:  # clear cash when there are too many entries
                print('CCBOND_BAR_MAP full, clear')
                CCBOND_BAR_MAP = {}
            print('reading from {}'.format(bar_path))
            bar_df = pd.read_csv(bar_path)
            CCBOND_BAR_MAP[bar_path] = bar_df
        if len(bar_df.loc[bar_df['Uid'] == uid]) == 0:
            print('!!! can not find data for {} at {}'.format(uid, minute_obj))
            return np.nan
        bar_df = bar_df.loc[bar_df['Uid'] == uid].iloc[:]
        bar_dfs.append(bar_df)
    big_bar_df = pd.concat(bar_dfs, axis=0)
    return big_bar_df


def read_ccbond_m1_bars_of_date(tgt_date, uid=None):
    begin_time, end_time = tgt_date + ' 09:30:00', tgt_date + ' 15:00:00'
    minutes = ct.trading_minutes_between(begin_time, end_time)
    bar_dfs = []
    for min_str in minutes:
        bar_path = '/data/hft/data_v1/dyndata/alpha_service/ccbond_bar_v5-1m/{}/{}/{}/{}.csv'.format(
            *tgt_date.split('-'), min_str.split(' ')[-1])
        # print('reading {}'.format(bar_path))
        bar_df = pd.read_csv(bar_path)
        bar_dfs.append(bar_df)
    big_bar_df = pd.concat(bar_dfs, axis=0)
    if uid is not None:
        big_bar_df = big_bar_df.loc[big_bar_df['Uid'] == uid]
    # print('got minute bars:\n{}'.format(big_bar_df))
    return big_bar_df


def write_ccbond_m1_factor_values(tgt_date, factor_name, df, include_index=False):
    dest_dir = '/data/current-rnd/data/cf/ws_ccbond/data/factor_values_m1/{}/{}/{}'.format(*tgt_date.split('-'))
    os.makedirs(dest_dir, exist_ok=True)
    dest_path = os.path.join(dest_dir, '{}.csv'.format(factor_name))

    df.to_csv(dest_path, index=include_index)
    logger.info('written {} values to {}'.format(factor_name, dest_path))

def read_ccbond_m1_factors_between(factor_name, begin_date, end_date):
    day_x_dfs = []
    for td in ct.trading_dates_between(begin_date, end_date):
        td = cu.dashed_date_str(td)
        x_path = '/data/current-rnd/data/cf/ws_ccbond/data/factor_values_m1/{}/{}/{}/{}.csv'.format(
            *td.split('-'), factor_name)
        day_x_df = pd.read_csv(x_path)
        day_x_dfs.append(day_x_df)
    x_df = pd.concat(day_x_dfs, axis=0)
    return x_df

def read_ccbond_Y_between(y_name, begin_date, end_date):
    day_y_dfs = []
    for td in ct.trading_dates_between(begin_date, end_date):
        td = cu.dashed_date_str(td)

        y_path = '/data/current-rnd/data/cf/ws_ccbond/data/Y_m1/{}/{}/{}/{}.csv'.format(*td.split('-'), y_name)
        # logger.info('reading Y from {}'.format(y_path))
        day_y_df = pd.read_csv(y_path)
        day_y_dfs.append(day_y_df)
    y_df = pd.concat(day_y_dfs, axis=0)
    return y_df

def read_ccbond_mold_df(begin_date, end_date, mold_name):
    mold_df = pd.read_csv('/data/current-rnd/data/cf/ws_ccbond/data/factor_values_m1/merged/{}.csv'.format(mold_name))
    mold_df = mold_df.loc[(mold_df['DateTime_R'] >= begin_date + ' 00:00:00') &
                          (mold_df['DateTime_R'] <= end_date + ' 23:00:00')]
    return mold_df



if __name__ == '__main__':
    # print('bond value: \n{}'.format(get_bond_value('127057.SZ', '2022-06-08')))
    # print('\n')
    # print('bond value: \n{}'.format(get_bond_value('113016.SH', '2022-06-08')))
    # print('\n')
    # print('bond value: \n{}'.format(get_bond_value('128015.SZ', '2021-09-02')))

    # print(get_corp_rate('2022-06-01', 'AAA', 1))

    # get_dolvol_and_vwap()
    # print(get_volatility('132009-SH-ccbond', ct.trading_minutes_between('2021-11-04 09:30:00', '2021-11-04 10:00:00')))
    # get_minute_bars_for_single_ccbond('132009-SH-ccbond', ct.trading_minutes_between('2021-11-04 09:30:00', '2021-11-04 10:00:00'))
    # get_hurst_exponent('127007-SZ-ccbond', ct.trading_minutes_between('2021-12-27 09:30:00', '2021-12-27 10:00:00'))
    # read_ccbond_m1_bars_of_date('128070-SZ-ccbond', '2021-11-16')
    read_ccbond_m1_bars('128070-SZ-ccbond', ['2021-11-16 09:30:00', '2021-11-16 15:00:00'])
