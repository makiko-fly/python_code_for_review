from abc import abstractmethod
import pandas as pd
from xfactor.runner.Database import Database
import copy
import os
import sys
import logging
from imp import reload
from limit_config import limit_config
from limit_config import overwrite_h5
reload(logging)
LOG_FILENAME = limit_config["log"]
logging.basicConfig(filename=LOG_FILENAME, filemode="w", level=logging.WARNING,
                    format='%(asctime)s - %(levelname)s - %(message)s')


class BaseFactor(object):
    depend_data = []
    depend_factors = []
    depend_nonfactors = []

    factor_type = "DAY"
    fix_times = ["1000", "1030", "1100", "1300", "1330", "1400", "1430"]
    lag = 0
    minute_lag = None
    financial_lag = 0
    reform_window = 1

    def __init__(self, **args):
        for k, v in args.items():
            setattr(self, k, v)
        if self.minute_lag is None:
            self.minute_lag = self.lag

    @classmethod
    def get_factor_class_name(cls):
        return cls.__name__

    def __update_single_database(self, database, single_database, data_info, day_start_date, minute_start_date,
                                 financial_start_date, end_datetime, financial_end_datetime, depend_type,
                                 fix_time="1500"):
        depend_type_data_list = eval("self." + depend_type)
        wind_str = "WIND_"
        gogoal_str = "SUNTIME_"
        h5str = "h5_"
        for depend_type_data in depend_type_data_list:
            if data_info[depend_type][depend_type_data] == "DAY":
                if fix_time == "1500":
                    value = eval('database.{}'.format(depend_type))[depend_type_data].loc[day_start_date: end_datetime]
                else:
                    # 如果是fix因子，只能取到前一天的日频数据
                    # value = eval('database.{}'.format(depend_type))[depend_type_data].loc[
                    #         day_start_date: end_datetime].iloc[:-1, ]
                    value = eval('database.{}'.format(depend_type))[depend_type_data].loc[day_start_date: end_datetime]
            elif data_info[depend_type][depend_type_data] == "MINUTE":
                if fix_time == "1500":
                    df = eval('database.{}'.format(depend_type))[depend_type_data]
                    value = df.loc[minute_start_date + "0925": end_datetime + fix_time]
                else:
                    df = eval('database.{}'.format(depend_type))[depend_type_data]
                    value = df.loc[minute_start_date + "0925": end_datetime + fix_time].iloc[:-1]
            elif data_info[depend_type][depend_type_data] in ["{}FINANCIAL".format(wind_str),
                                                              "{}FINANCIAL".format(h5str),
                                                              "{}FINANCIAL".format(gogoal_str)]:
                value = self.__get_singleh5_value(database, depend_type, depend_type_data, financial_start_date,
                                                  financial_end_datetime)
            else:
                raise Exception("data type only DAY or MINUTE!")
            single_database.update(depend_type, depend_type_data, value)


    def __get_singleh5_value(self, database, depend_type, depend_type_data, financial_start_date,
                             financial_end_datetime):
        table_name = depend_type_data.split(".")[1]
        if table_name in overwrite_h5:
            value = eval('database.{}'.format(depend_type))[depend_type_data]
        else:
            df = eval('database.{}'.format(depend_type))[depend_type_data]
            if "htsc_date" in df.columns:
                value = df.query(
                    '{} < {} & {} >= {}'.format("htsc_date", financial_end_datetime,
                                                "htsc_date", financial_start_date))
                value.drop(["htsc_date"], axis=1, inplace=True)
            else:
                value = df[(df.index.get_level_values(0) < financial_end_datetime) & (
                        df.index.get_level_values(0) >= financial_start_date)]
        return value

    def calc(self, database, data_info, calc_datetime_list, full_datetime_list, for_temp, with_temp):
        result = {}
        factor_name = self.get_factor_class_name()
        if with_temp and factor_name in os.listdir(with_temp):
            start_index = full_datetime_list.index(calc_datetime_list[0])
            end_index = start_index + len(calc_datetime_list)
        else:
            start_index = full_datetime_list.index(calc_datetime_list[0]) - max(self.reform_window - 1, 0)
            end_index = start_index + len(calc_datetime_list) + max(self.reform_window - 1, 0)

        if self.factor_type == "DAY":
            result_dict = {}
            end_index_t = end_index
            if with_temp and factor_name + ".pkl" in os.listdir(with_temp):
                start_index = end_index - 1
            if for_temp and self.reform_window > 1:
                end_index_t = end_index - 1
            for index in range(start_index, end_index_t):
                single_database = Database()
                day_start_date = full_datetime_list[index - self.lag]
                minute_start_date = full_datetime_list[index - self.minute_lag]
                financial_start_date = full_datetime_list[index - self.financial_lag]
                # end_datetime = full_datetime_list[index]
                # financial_end_datetime = full_datetime_list[index + 1]
                end_datetime = '20210530'
                financial_end_datetime = '20210530'
                if database.depend_data:
                    self.__update_single_database(database, single_database, data_info, day_start_date,
                                                  minute_start_date, financial_start_date, end_datetime,
                                                  financial_end_datetime, "depend_data", "1500")
                if database.depend_factors:
                    self.__update_single_database(database, single_database, data_info, day_start_date,
                                                  minute_start_date, financial_start_date, end_datetime,
                                                  financial_end_datetime, "depend_factors", "1500")
                if database.depend_nonfactors:
                    self.__update_single_database(database, single_database, data_info, day_start_date,
                                                  minute_start_date, financial_start_date, end_datetime,
                                                  financial_end_datetime, "depend_nonfactors", "1500")

                single_database_copy = copy.deepcopy(single_database)
                single_res = self.calc_single(single_database_copy)
                result_dict[end_datetime] = single_res
            result_df = pd.DataFrame.from_dict(result_dict, orient="index")

            if for_temp and self.reform_window > 1:
                result.update({factor_name: result_df})
            elif with_temp and factor_name + ".pkl" in os.listdir(with_temp):
                temp_res = pd.read_pickle("{}/{}.pkl".format(with_temp, factor_name))
                result_df = temp_res.append(result_df)
                result_df = self.reform(result_df).iloc[max(self.reform_window - 1, 0):]
                result.update({factor_name: result_df})
            else:
                result_df = self.reform(result_df).iloc[max(self.reform_window - 1, 0):]
                tmp = result_df.isna().all(axis=1)
                tmp = tmp[tmp == True]
                if len(tmp) > 0:
                    for i in tmp.index:
                        logging.warning(factor_name + " all NAN in " + i)
                result.update({factor_name: result_df})
            return result

        elif self.factor_type == "FIX":
            for fix_time in self.fix_times:
                result_dict = {}
                current_factor_name = "Fix" + fix_time + "_" + factor_name
                end_index_t = end_index
                if with_temp and current_factor_name + ".pkl" in os.listdir(with_temp):
                    start_index = end_index - 1
                if for_temp and self.reform_window > 1:
                    end_index_t = end_index - 1
                for index in range(start_index, end_index_t):
                    single_database = Database()
                    day_start_date = full_datetime_list[index - self.lag]
                    minute_start_date = full_datetime_list[index - self.minute_lag]
                    financial_start_date = full_datetime_list[index - self.financial_lag]
                    # end_datetime = full_datetime_list[index]
                    end_datetime = '20210522'
                    # financial_end_datetime = full_datetime_list[index + 1]
                    financial_end_datetime = end_datetime

                    if database.depend_data:
                        self.__update_single_database(database, single_database, data_info, day_start_date,
                                                      minute_start_date, financial_start_date, end_datetime,
                                                      financial_end_datetime, "depend_data", fix_time)
                    if database.depend_factors:
                        self.__update_single_database(database, single_database, data_info, day_start_date,
                                                      minute_start_date, financial_start_date, end_datetime,
                                                      financial_end_datetime, "depend_factors", fix_time)
                    if database.depend_nonfactors:
                        self.__update_single_database(database, single_database, data_info, day_start_date,
                                                      minute_start_date, financial_start_date, end_datetime,
                                                      financial_end_datetime, "depend_nonfactors", fix_time)
                    single_database_copy = copy.deepcopy(single_database)
                    single_res = self.calc_single(single_database_copy)
                    result_dict[end_datetime + fix_time] = single_res
                result_df = pd.DataFrame.from_dict(result_dict, orient="index")
                result_df.index = list(map(lambda x: x[:-4], result_df.index))
                if for_temp:
                    if self.reform_window > 1:
                        result.update({current_factor_name: result_df})
                    else:
                        break
                elif with_temp and current_factor_name + ".pkl" in os.listdir(with_temp):
                    temp_res = pd.read_pickle("{}/{}.pkl".format(with_temp, current_factor_name))
                    result_df = temp_res.append(result_df)
                    result_df = self.reform(result_df).iloc[max(self.reform_window - 1, 0):]
                    result.update({current_factor_name: result_df})
                else:
                    result_df = self.reform(result_df).iloc[max(self.reform_window - 1, 0):]
                    tmp = result_df.isna().all(axis=1)
                    tmp = tmp[tmp == True]
                    if len(tmp) > 0:
                        for i in tmp.index:
                            logging.warning(current_factor_name + " all NAN in " + i)
                    result.update({current_factor_name: result_df})
            return result

    @abstractmethod
    def calc_single(self, single_database):
        return None

    def reform(self, temp_result):
        return temp_result
