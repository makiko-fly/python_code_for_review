import os
import sys
import datetime as dt
import pathlib
import threading
import numpy as np
import pandas as pd

CUR_DIR = str(pathlib.Path(__file__).parent.absolute())
sys.path.insert(0, CUR_DIR)
import cf_util

TRADING_DATES = []
TRADING_DATES_LOCK = threading.Lock()

def read_trade_date_file():
    date_path = '/data/hft/data_v1/hftdata/WIND/static/entire/trade_date.csv'
    if not os.path.exists(date_path): # not able to access ceph, use local resource folder
        print('== not able to access ceph, use local resource folder')
        date_path = CUR_DIR + '/res/trade_date.csv'
    # read from file
    with open(date_path) as date_file:
        for line in date_file:
            line = line.strip()
            if not line:
                continue
            TRADING_DATES.append(line)

def get_index_of_date(date=None, side='left'):
    TRADING_DATES_LOCK.acquire()
    try:
        if len(TRADING_DATES) == 0:
            read_trade_date_file()
        idx = np.searchsorted(TRADING_DATES, date, side=side)
        assert idx > 0, 'found idx {} is first of list, invalid'.format(idx)
        assert idx < len(TRADING_DATES)-1, 'found idx {} is last of list, invalid'.format(idx)
        return idx
    finally:
        TRADING_DATES_LOCK.release()


def prev_trading_date(date=None, days=1):
    date = cf_util.compact_date_str(date) 
    return TRADING_DATES[get_index_of_date(date, 'left')-days]

def prev_date(date):    
    if isinstance(date, str):
        date = cf_util.date_from_str(date)
    return date - dt.timedelta(days=1)
   
def next_trading_date(date=None, days=1):
    date = cf_util.compact_date_str(date) 
    return TRADING_DATES[get_index_of_date(date, 'right')+days-1]

def next_date(date):
    if isinstance(date, str):
        date = cf_util.date_from_str(date)
    return date + dt.timedelta(days=1)

def is_trading_date(date=None):
    date = cf_util.compact_date_str(date) 
    return TRADING_DATES[get_index_of_date(date, 'left')] == date

def trading_dates_between(begin=None, end=None, asset_group='CN'):
    assert begin and end, 'begin and end cannot be none'
    if asset_group in ['CN', 'commodity']:
        begin = cf_util.compact_date_str(begin)
        end = cf_util.compact_date_str(end)
        # adjust end date
        if not is_trading_date(end):
            end = prev_trading_date(end)
        begin_idx = get_index_of_date(begin, 'left')
        end_idx = get_index_of_date(end, 'left')
        return TRADING_DATES[begin_idx:end_idx+1]
    elif asset_group.upper() == 'BLOCKCHAIN':
        return [x.strftime('%Y-%m-%d') for x in pd.date_range(begin, end)]
    else:
        raise Exception('not supported asset group')

def trade_date_from_order_time(order_time):
    '''get book date from order time, should accommodate for night trading, for example:
        order_time: 2013-07-10 09:37:00 -> book date: 2013-07-10
        order_time: 2013-07-10 21:37:00 -> book date: next trading date
        order_time: 2013-07-10 01:37:00 -> book date: prev day's next trading date
    '''
    date, time = order_time.split(' ')
    if time > '18:00:00':
        return cf_util.dashed_date_str(next_trading_date(date))
    elif time < '04:00:00':
        return cf_util.dashed_date_str(next_trading_date(prev_date(date)))
    return date 


if __name__ == '__main__':
    ret = prev_trading_date('20200222')
    print('ret: ', ret)

