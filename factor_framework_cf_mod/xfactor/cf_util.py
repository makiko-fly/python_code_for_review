import os
import datetime as dt
import re
import pathlib
import importlib

import pandas as pd

def num_of_chinese_chars(text):
    return len(re.findall('[\u4e00-\u9FFF]', text))



def compact_date_str(date=None):
    if isinstance(date, int):  # sample: 20200204 treated as int
        date = str(date)
    if isinstance(date, str):  # date sample: 20200204 or 2020-02-04
        if '-' in date:
            date = date.replace('-', '')
        assert len(date) == 8, 'date {} is not valid'.format(date)
        return date
    if not date:
        date = dt.datetime.now()
    return date.strftime('%Y%m%d')

def dashed_date_str(date=None):
    if isinstance(date, int):  # sample: 20200204 treated as int
        date = str(date)

    if isinstance(date, str):  # date sample: 20200204, 2020-02-04 DateTime(2020, 2, 4)
        if '-' in date:
            return date
        date = dt.datetime.strptime(date, '%Y%m%d')
    if not date:
        date = dt.datetime.now()
    return date.strftime('%Y-%m-%d')

def date_from_str(v):
    if not isinstance(v, str):
        raise Exception('date from str, value should be str')
    if '-' in v:  # sample: 2020-02-04
        return dt.datetime.strptime(v, '%Y-%m-%d')
    elif len(v) == 8: # sample: 20200204
        return dt.datetime.strptime(v, '%Y%m%d')
    else:
        raise Exception('date_from_str, unsupported format: {}'.format(v))

def get_min_date(dir_path):
    min_year = min([name for name in os.listdir(dir_path) if name.isdigit()])
    assert len(min_year) == 4, 'should not happen'
    min_month = min(os.listdir(dir_path + '/' + min_year))
    assert len(min_month) == 2, 'should not happen'
    min_day = min(os.listdir(dir_path + '/' + min_year + '/' + min_month))
    assert len(min_day) == 2, 'should not happen'
    return min_year + '-' + min_month + '-' + min_day
def get_max_date(dir_path):
    max_year = max([name for name in os.listdir(dir_path) if name.isdigit()])
    assert len(max_year) == 4, 'should not happen'
    max_month = max(os.listdir(dir_path + '/' + max_year))
    assert len(max_month) == 2, 'should not happen'
    max_day = max(os.listdir(dir_path + '/' + max_year + '/' + max_month))
    assert len(max_day) == 2, 'should not happen'
    return max_year + '-' + max_month + '-' + max_day    

def to_semiyear(date):
    date = compact_date_str(date)
    year, month = date[:4], date[4:6]
    semi = 'H1' if month <= '06' else 'H2'
    return year + '-' + semi
def to_MicroSecondSinceEpoch(obj):
    if isinstance(obj, pd.Series):
        return pd.to_datetime(obj).dt.strftime('%s%f')
    datetime = obj
    if isinstance(datetime, str):
        datetime = pd.to_datetime(datetime)
    return datetime.strftime('%s%f')

# ======== numpy and pandas related
def print_entire_df(df):
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', None):
        print(df)
def move_cols_to_front(df, cols_to_move):
    df = df[cols_to_move + [col for col in df.columns if col not in cols_to_move]]
    return df
def move_cols_to_back(df, cols_to_move):
    df = df[[col for col in df.columns if col not in cols_to_move] + cols_to_move] 
    return df
def drop_nans(df, cols=None):
    if cols is None:
        cols = list(df.columns)
    if isinstance(cols, str):
        cols = [cols]
    ret_df = df.loc[~df[cols].isna().any(axis=1)]
    return ret_df

def import_module_from_path(module_name, path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]
