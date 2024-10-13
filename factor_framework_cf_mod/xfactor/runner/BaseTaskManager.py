import sys
import pathlib

import xfactor.FactorUtil as FactorUtil
from abc import abstractmethod
# from xquant.factordata import FactorData
import pandas as pd
import xfactor.trading_date as td

CUR_DIR = str(pathlib.Path(__file__).parent.absolute())

class BaseTaskManager(object):
    max_date_num_per_task = 100
    max_factor_num_per_task = 1
    # fa = FactorData()

    def __init__(self, factor_name_list, start_date, end_date, fix_config, input_factor_lib=None):
        self.factor_class_list = FactorUtil.get_factor_class_list(factor_name_list)
        self.start_date = start_date
        self.end_date = end_date
        self.input_factor_lib = input_factor_lib
        self.fix_config = fix_config
        # 根据最大计算时间跨度，将计算时间进行分组
        self.calc_time_groups = self.split_calc_datetime_into_group()
        self.calc_factor_groups = self.split_calc_factor_into_group()
        self.full_datetime_list = self.__get_full_datetime_list()
        self.stock_list = self.__get_stock_list()

    @abstractmethod
    def split_calc_datetime_into_group(self):
        return None

    @abstractmethod
    def split_calc_factor_into_group(self):
        return None

    def __get_stock_list(self):
        universe_df = pd.read_csv(CUR_DIR + '/../tmp_data/daily_20210521.csv')
        # minute_stock_list = pd.read_csv("/data/user/666889/Apollo/AlphaDataBase/CompleteStockList.csv")[
        #     "Stock_code"].tolist()
        # return minute_stock_list
        return list(universe_df['Uid'])

    # 获取全部交易日列表，包括lag部分的日期
    def __get_full_datetime_list(self):
        max_data_exceed = FactorUtil.get_max_data_exceed(self.factor_class_list)
        shift_start_date = td.prev_trading_date(self.start_date, days=max_data_exceed)
        full_datetime_list = td.trading_dates_between(shift_start_date, self.end_date)
        # start_date2 = self.start_date
        # end_date2 = self.fa.tradingday(self.end_date, 2)[-1]
        # if max_data_exceed > 0:
        #     start_date2 = int(self.fa.tradingday(self.start_date, -(max_data_exceed + 1))[0])
        # full_datetime_list = self.fa.tradingday(start_date2, end_date2)
        return full_datetime_list

    # 生成task
    def generate_task(self):
        task_list = []
        for calc_time_group in self.calc_time_groups:
            for calc_factor_group in self.calc_factor_groups:
                task_list.append({
                    "factor_class_list": calc_factor_group,
                    "fix_config": self.fix_config,
                    "stock_list": self.stock_list,
                    "calc_time_list": calc_time_group,
                    "full_datetime_list": self.full_datetime_list,
                    "input_factor_lib": self.input_factor_lib
                })
        return task_list

