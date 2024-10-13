import numpy as np
import pandas as pd
import datetime as dt
# from xquant.factordata import FactorData
import copy

np.seterr(invalid='ignore')


def minute_data_transform(depend_data, operation=["merge", "merge"]):
    data_item_map = [(i.split('.')[2][:-7], i) for i in depend_data.keys() if
                     "Basic_factor" in i and i.split(".")[2][-7:] == "_minute"]
    for oper in operation:
        if not (oper == "" or oper.startswith("merge") or oper.startswith("drop")):
            raise TypeError
    for item, key in data_item_map:
        trans_data = __minute_single_data_transform(item, depend_data[key], operation)
        depend_data.update({key: trans_data})


def __minute_single_data_transform(item, df, operation):
    df.index = pd.to_datetime(df.index)  # make sure df.index is of class DateTime
    df_index = list(df.index)
    each_day_index_list = []
    one_day_index = []
    for index in df_index:
        if index.hour == 9 and index.minute == 25:
            one_day_index = [index]
        elif index.hour == 15 and index.minute == 00:
            one_day_index.append(index)
            each_day_index_list.append(one_day_index)
        else:
            one_day_index.append(index)
    if index.hour != 15:
        each_day_index_list.append(one_day_index)

    for each_day_index in each_day_index_list:
        if operation[0].startswith("drop"):
            drop_num = operation[0][len("drop"):]
            drop_num = 1 if drop_num == "" else int(drop_num)
            drop_index_list = [each_day_index[i] for i in range(drop_num)]
            for drop_index in drop_index_list:
                df.drop(index=drop_index)
        elif operation[0].startswith("merge"):
            merge_num = operation[0][len("merge"):]
            merge_num = 1 if merge_num == "" else int(merge_num)
            merge_index_list = [each_day_index[i] for i in range(merge_num)]
            result_index = each_day_index[merge_num]
            for merge_index in merge_index_list:
                if item in ['volume', 'amt']:
                    df.loc[result_index] = df.loc[merge_index].values + df.loc[result_index].values
                elif item == "open":
                    df.loc[result_index] = df.loc[merge_index_list[0]].values
                elif item == "low":
                    df.loc[result_index] = np.minimum(df.loc[merge_index].values, df.loc[result_index].values)
                df.drop(merge_index, axis=0, inplace=True)
        else:
            pass

        if each_day_index[-1].hour == 15 and each_day_index[-1].minute == 0:
            if operation[1].startswith("drop"):
                drop_num = operation[1][len("drop"):]
                drop_num = 1 if drop_num == "" else int(drop_num)
                drop_index_list = [each_day_index[-(i + 1)] for i in range(drop_num)]
                for drop_index in drop_index_list:
                    df.drop(drop_index, axis=0, inplace=True)
            elif operation[1].startswith("merge"):
                merge_num = operation[1][len("merge"):]
                merge_num = 1 if merge_num == "" else int(merge_num)
                merge_index_list = [each_day_index[-(i + 1)] for i in range(merge_num)]
                result_index = each_day_index[-(merge_num + 1)]
                for merge_index in merge_index_list:
                    if item in ['volume', 'amt']:
                        df.loc[result_index] = df.loc[merge_index].values + df.loc[result_index].values
                    elif item == "close":
                        df.loc[result_index] = df.loc[merge_index].values
                    elif item == "low":
                        df.loc[result_index] = np.minimum(df.loc[merge_index].values, df.loc[result_index].values)
                    df.drop(merge_index, axis=0, inplace=True)
            else:
                pass
    return df


def min_forward_adj(df, date=None):
    #### 给分钟级别的因子进行复权 #########
    #### 如果传入dataframe,则不需要传入时间######
    #### 如果传入Series,则需要传入时间######
    fa = FactorData()
    df = copy.deepcopy(df)
    if date == None:
        stock_list = df.columns.tolist()
        startdate = df.index[0]
        enddate = df.index[-1]
        start_date = startdate.year * 10000 + startdate.month * 100 + startdate.day
        end_date = enddate.year * 10000 + enddate.month * 100 + enddate.day

        df_adjfactor = pd.read_hdf('/data/group/800080/Apollo/AlphaDataBase/Data_adjfactor.h5', '/factor')
        df_adjfactor = df_adjfactor.loc[start_date:end_date]
        df_adjfactor.index = list(map(lambda x: dt.datetime.strptime(str(x), "%Y%m%d"), df_adjfactor.index))
        df_adjfactor = df_adjfactor.reindex(columns=stock_list)

        # df_adjfactor = \
        #     fa.get_factor_value("Basic_factor", stock_list, fa.tradingday(start_date, end_date), ["adjfactor"])[
        #         "adjfactor"].unstack()
        # df_adjfactor.index = list(map(lambda x: dt.datetime.strptime(x, "%Y%m%d"), df_adjfactor.index))
        # 将分钟行情的df设为双重索引，其第1重索引为日期
        df['date'] = df.index.date
        df.index.name = 'datetime'
        df = df.reset_index()
        df = df.set_index(['date', 'datetime'])
        # 将分钟行情的df与复权因子按日期相乘
        df = df.multiply(df_adjfactor, level=0)
        # 去除分钟行情的df的双重索引中日期的索引，仅保留datetime这一索引
        df = df.reset_index()
        df = df.set_index('datetime')
        df = df.reindex(columns=stock_list)
    else:
        stock_list = df.index.tolist()
        start_date = date.year * 10000 + date.month * 100 + date.day

        # 暂时用666889中的数据代替get_factor_value
        df_adjfactor = pd.read_hdf('/data/group/800080/Apollo/AlphaDataBase/Data_adjfactor.h5', '/factor')
        df_adjfactor = df_adjfactor.loc[start_date].to_frame().T
        df_adjfactor.index = list(map(lambda x: dt.datetime.strptime(str(x), "%Y%m%d"), df_adjfactor.index))
        df_adjfactor = df_adjfactor.reindex(columns=stock_list)

        # end_date = start_date
        # df_adjfactor = \
        #     fa.get_factor_value("Basic_factor", stock_list, fa.tradingday(start_date, end_date), ["adjfactor"])[
        #         "adjfactor"].unstack()
        # df_adjfactor.index = list(map(lambda x: dt.datetime.strptime(x, "%Y%m%d"), df_adjfactor.index))

        df_adjfactor = df_adjfactor.iloc[-1, :]
        df = df * df_adjfactor
    return df