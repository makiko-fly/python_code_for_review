import math
import os
import sys
import datetime as dt
import pathlib
import threading
import numpy as np
import pandas as pd


CUR_DIR = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
if CUR_DIR not in sys.path:
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
   
def next_trading_date(date=None, days=1, dashed=False):
    date = cf_util.compact_date_str(date)
    ret_date = TRADING_DATES[get_index_of_date(date, 'right')+days-1]
    if dashed:
        ret_date = cf_util.dashed_date_str(ret_date)
    return ret_date

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

def trading_years_between(begin_date, end_date):
    tds = [cf_util.dashed_date_str(td) for td in trading_dates_between(begin_date, end_date)]
    years = sorted(list(set([td[:4] for td in tds])))
    return years

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

def num_of_calendar_days(begin_date, end_date):
    begin_date_s, end_date_s = cf_util.dashed_date_str(begin_date), cf_util.dashed_date_str(end_date)
    begin_date_o, end_date_o = dt.datetime.strptime(begin_date_s, '%Y-%m-%d'), dt.datetime.strptime(end_date_s, '%Y-%m-%d')
    ret_days = (end_date_o - begin_date_o).days
    assert ret_days >= 0, 'invalid begin_date {} and end_date {}'.format(begin_date, end_date)
    return ret_days





TRADING_MINUTES_R = ['09:30:00', '09:31:00', '09:32:00', '09:33:00', '09:34:00', '09:35:00', '09:36:00', '09:37:00',
                     '09:38:00', '09:39:00', '09:40:00', '09:41:00', '09:42:00', '09:43:00', '09:44:00',
                     '09:45:00', '09:46:00', '09:47:00', '09:48:00', '09:49:00', '09:50:00', '09:51:00',
                     '09:52:00', '09:53:00', '09:54:00', '09:55:00', '09:56:00', '09:57:00', '09:58:00',
                     '09:59:00', '10:00:00', '10:01:00', '10:02:00', '10:03:00', '10:04:00', '10:05:00',
                     '10:06:00', '10:07:00', '10:08:00', '10:09:00', '10:10:00', '10:11:00', '10:12:00',
                     '10:13:00', '10:14:00', '10:15:00', '10:16:00', '10:17:00', '10:18:00', '10:19:00',
                     '10:20:00', '10:21:00', '10:22:00', '10:23:00', '10:24:00', '10:25:00', '10:26:00',
                     '10:27:00', '10:28:00', '10:29:00', '10:30:00', '10:31:00', '10:32:00', '10:33:00',
                     '10:34:00', '10:35:00', '10:36:00', '10:37:00', '10:38:00', '10:39:00', '10:40:00',
                     '10:41:00', '10:42:00', '10:43:00', '10:44:00', '10:45:00', '10:46:00', '10:47:00',
                     '10:48:00', '10:49:00', '10:50:00', '10:51:00', '10:52:00', '10:53:00', '10:54:00',
                     '10:55:00', '10:56:00', '10:57:00', '10:58:00', '10:59:00', '11:00:00', '11:01:00',
                     '11:02:00', '11:03:00', '11:04:00', '11:05:00', '11:06:00', '11:07:00', '11:08:00',
                     '11:09:00', '11:10:00', '11:11:00', '11:12:00', '11:13:00', '11:14:00', '11:15:00',
                     '11:16:00', '11:17:00', '11:18:00', '11:19:00', '11:20:00', '11:21:00', '11:22:00',
                     '11:23:00', '11:24:00', '11:25:00', '11:26:00', '11:27:00', '11:28:00', '11:29:00',
                     '11:30:00', '13:01:00', '13:02:00', '13:03:00', '13:04:00', '13:05:00', '13:06:00',
                     '13:07:00', '13:08:00', '13:09:00', '13:10:00', '13:11:00', '13:12:00', '13:13:00',
                     '13:14:00', '13:15:00', '13:16:00', '13:17:00', '13:18:00', '13:19:00', '13:20:00',
                     '13:21:00', '13:22:00', '13:23:00', '13:24:00', '13:25:00', '13:26:00', '13:27:00',
                     '13:28:00', '13:29:00', '13:30:00', '13:31:00', '13:32:00', '13:33:00', '13:34:00',
                     '13:35:00', '13:36:00', '13:37:00', '13:38:00', '13:39:00', '13:40:00', '13:41:00',
                     '13:42:00', '13:43:00', '13:44:00', '13:45:00', '13:46:00', '13:47:00', '13:48:00',
                     '13:49:00', '13:50:00', '13:51:00', '13:52:00', '13:53:00', '13:54:00', '13:55:00',
                     '13:56:00', '13:57:00', '13:58:00', '13:59:00', '14:00:00', '14:01:00', '14:02:00',
                     '14:03:00', '14:04:00', '14:05:00', '14:06:00', '14:07:00', '14:08:00', '14:09:00',
                     '14:10:00', '14:11:00', '14:12:00', '14:13:00', '14:14:00', '14:15:00', '14:16:00',
                     '14:17:00', '14:18:00', '14:19:00', '14:20:00', '14:21:00', '14:22:00', '14:23:00',
                     '14:24:00', '14:25:00', '14:26:00', '14:27:00', '14:28:00', '14:29:00', '14:30:00',
                     '14:31:00', '14:32:00', '14:33:00', '14:34:00', '14:35:00', '14:36:00', '14:37:00',
                     '14:38:00', '14:39:00', '14:40:00', '14:41:00', '14:42:00', '14:43:00', '14:44:00',
                     '14:45:00', '14:46:00', '14:47:00', '14:48:00', '14:49:00', '14:50:00', '14:51:00',
                     '14:52:00', '14:53:00', '14:54:00', '14:55:00', '14:56:00', '14:57:00', '14:58:00',
                     '14:59:00', '15:00:00']

def _get_index_of_minute(min_str):
    # simple check, can be improved
    assert len(min_str) == 8 and ':' in min_str, 'min_str {} format error'.format(min_str)
    if min_str < '09:30:00':
        min_str = '09:30:00'
    elif '11:30:00' < min_str < '13:01:00':
        min_str = '13:01:00'
    elif min_str > '15:00:00':
        min_str = '15:00:00'
    return TRADING_MINUTES_R.index(min_str)

def next_trading_minute(minute_obj, minutes=1, overnight=False, align='R'):
    minute_obj = pd.to_datetime(minute_obj)
    date_str = cf_util.dashed_date_str(minute_obj)
    min_str = minute_obj.strftime('%H:%M:%S')
    idx = _get_index_of_minute(min_str)
    tgt_idx = idx + minutes

    if overnight:  # can span several days
        look_forward_days = int(math.ceil(minutes / 240))
        end_minute = cf_util.dashed_date_str(next_trading_date(date_str, days=look_forward_days)) + ' 15:00:00'
        trading_minutes = trading_minutes_between(minute_obj, end_minute)
        return trading_minutes[minutes]
    else:
        if tgt_idx > len(TRADING_MINUTES_R) - 1:
            tgt_idx = len(TRADING_MINUTES_R) - 1
        return date_str + ' ' + TRADING_MINUTES_R[tgt_idx]

def trading_minutes_between(begin, end, include_auction=False, include_end=True, asset_group=None):
    first_minute = '09:30:00' if include_auction else '09:31:00'
    # move begin and end to trade dates if necessary
    if not is_trading_date(begin):
        begin = cf_util.dashed_date_str(next_trading_date(begin)) + ' ' + first_minute
    if not is_trading_date(end):
        end = cf_util.dashed_date_str(prev_trading_date(end)) + ' 15:00:00'
    begin, end = pd.to_datetime(begin), pd.to_datetime(end)
    if cf_util.dashed_date_str(begin) != cf_util.dashed_date_str(end):  # not the same trading day
        assert len(trading_dates_between(begin, end)) < 30, 'too many trading days between {} and {}'.format(begin, end)

        ret_min_strs = []
        for td in trading_dates_between(begin, end):
            if cf_util.dashed_date_str(td) == cf_util.dashed_date_str(begin):
                ret_min_strs += trading_minutes_between(begin, cf_util.dashed_date_str(td) + ' 15:00:00',
                                                        include_auction=include_auction, include_end=include_end)
            elif cf_util.dashed_date_str(td) == cf_util.dashed_date_str(end):
                ret_min_strs += trading_minutes_between(cf_util.dashed_date_str(td) + ' ' + first_minute, end,
                                                        include_auction=include_auction, include_end=include_end)
            else:
                ret_min_strs += trading_minutes_between(cf_util.dashed_date_str(td) + ' ' + first_minute,
                                                        cf_util.dashed_date_str(td) + ' 15:00:00',
                                                        include_auction=include_auction, include_end=include_end)
        return ret_min_strs
    else:  # same trade date
        begin_idx = _get_index_of_minute(begin.strftime('%H:%M:%S'))
        end_idx = _get_index_of_minute(end.strftime('%H:%M:%S'))
        if include_end:
            return [cf_util.dashed_date_str(begin) + ' ' + m for m in TRADING_MINUTES_R[begin_idx:end_idx+1]]
        else:
            return [cf_util.dashed_date_str(begin) + ' ' + m for m in TRADING_MINUTES_R[begin_idx:end_idx]]



if __name__ == '__main__':
    # ret = prev_trading_date('20200222')
    # print('ret: ', ret)
    # ret = num_of_calendar_days('2022-06-01', '2022-05-01')
    # print(ret)

    # dt_obj = pd.to_datetime('2022-07-13 09:30:00')
    # str_list = []
    # for i in range(500):
    #     dt_obj += dt.timedelta(minutes=1)
    #     if dt_obj.hour == 11 and dt_obj.minute > 30:
    #         continue
    #     elif dt_obj.hour == 12:
    #         continue
    #     elif dt_obj.hour >= 15:
    #         continue
    #     str_list.append(dt_obj.strftime('%H:%M:%S'))
    # str_list = ['\'{}\''.format(the_str) for the_str in str_list]
    # print(','.join(str_list))

    # print(get_next_trading_minute('2022-07-13 14:59:00', minutes=1))
    # print(trading_minutes_between('2022-07-11 11:45:00', '2022-07-13 11:30:00'))
    print(trading_minutes_between('2021-09-03 10:46:00', '2021-09-06 09:46:00', include_end=False))
    # print(next_trading_minute('2022-07-11 10:30:00', minutes=480, intraday=True))