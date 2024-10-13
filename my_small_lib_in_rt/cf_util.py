import datetime as dt
import pathlib

import pandas as pd



def compact_date_str(date=None):
    if not date:
        date = dt.datetime.now()
    else:
        date = pd.to_datetime(date)
    return date.strftime('%Y%m%d')
    # if isinstance(date, dt.datetime):
    #     return date.strftime('%Y%m%d')
    # if isinstance(date, int):  # sample: 20200204 treated as int
    #     date = str(date)
    # if isinstance(date, str):  # date sample: 20200204 or 2020-02-04
    #     if '-' in date:
    #         date = date.replace('-', '')
    #     assert len(date) == 8, 'date {} is not valid'.format(date)
    #     return date
    # if not date:
    #     date = dt.datetime.now()
    # return date.strftime('%Y%m%d')


def dashed_date_str(date=None):
    if not date:
        date = dt.datetime.now()
    else:
        date = pd.to_datetime(date)
    return date.strftime('%Y-%m-%d')
    # if isinstance(date, int):  # sample: 20200204 treated as int
    #     date = str(date)
    #
    # if isinstance(date, str):  # date sample: 20200204, 2020-02-04 DateTime(2020, 2, 4)
    #     if '-' in date:
    #         return date
    #     date = dt.datetime.strptime(date, '%Y%m%d')
    # if not date:
    #     date = dt.datetime.now()
    # return date.strftime('%Y-%m-%d')


def date_from_str(v):
    if not isinstance(v, str):
        raise Exception('date from str, value should be str')
    if '-' in v:  # sample: 2020-02-04
        return dt.datetime.strptime(v, '%Y-%m-%d')
    elif len(v) == 8:  # sample: 20200204
        return dt.datetime.strptime(v, '%Y%m%d')
    else:
        raise Exception('date_from_str, unsupported format: {}'.format(v))


def to_semiyear(date):
    date = compact_date_str(date)
    year, month = date[:4], date[4:6]
    semi = 'H1' if month <= '06' else 'H2'
    return year + '-' + semi


def print_entire_df(df):
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', None):
        print(df)


def move_cols_to_front(df, cols_to_move):
    if isinstance(cols_to_move, str):  # single column string
        cols_to_move = [cols_to_move]
    df = df[cols_to_move + [col for col in df.columns if col not in cols_to_move]]
    return df


def move_cols_to_back(df, cols_to_move):
    if isinstance(cols_to_move, str):  # single column string
        cols_to_move = [cols_to_move]
    df = df[[col for col in df.columns if col not in cols_to_move] + cols_to_move] 
    return df


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def drop_nans(df, cols=None):
    if cols is None:
        cols = list(df.columns)
    if isinstance(cols, str):
        cols = [cols]
    ret_df = df.loc[~df[cols].isna().any(axis=1)]
    return ret_df


def split_factor_names(factor_names):
    if not isinstance(factor_names, list):
        if ',' in factor_names:
            factor_names = factor_names.split(',')
        else:
            factor_names = [factor_names]
    return factor_names

def to_str_list(text):
    if not isinstance(text, list):
        if ',' in text:
            str_list = text.split(',')
        else:
            str_list = [text]
    else:
        str_list = text
    return str_list




def to_pandas_freq(freq):
    freq_map = {'m1': '1min', 'm5': '5min', 'm10': '10min', 'm15': '15min', 'm30': '30min', 'D1': '1D'}
    return freq if freq not in freq_map else freq_map[freq]

def my_factor_sort_key(fname):
    if fname.startswith('BR'):
        return int(fname.replace('BR_', ''))
    elif fname.startswith('CF'):
        return int('1' + fname.replace('CF_', '')) * 10  # prefix with '1' to avoid CF_0001 issue
    elif fname.startswith('NA_CF'):
        return int('99' + fname.replace('NA_CF_', '')) * 100
    elif fname.startswith('NA_BR'):
        return int('99' + fname.replace('NA_BR_', '')) * 10
    else:
        raise Exception('not yet supported factor: {}'.format(fname))

def remove_formula_spaces(formula_dict):
    single_spaced_dict = dict()
    import re
    for fname, formula in formula_dict.items():
        single_spaced_dict[fname] = re.sub(' +', ' ', formula)
    return single_spaced_dict
