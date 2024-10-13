# import ray
import sys
from xfactor.runner.Database import Database
import xfactor.FactorUtil as FactorUtil
# from xquant.factordata import FactorData
import pandas as pd
import datetime as dt
from h5data.IO import IO
import numpy as np
# from xquant.thirdpartydata.marketdata import MarketData
from limit_config import limit_config
from limit_config import overwrite_h5
from limit_config import mapping_h5
from multiprocessing import Pool
from itertools import repeat

# fa = FactorData()
today = dt.datetime.today().strftime("%Y%m%d")
wind_str = "WIND_"
gogoal_str = "SUNTIME_"
daily_local_path = limit_config["daily_path"]
minute_local_path = limit_config["minute_path"]
h5_local_path = limit_config["h5_path"]


# 获取h5表数据
def __get_fanancial_data(table_name, start_date, end_date, root_path, col=None):
    if table_name == "ETC_CHINA_STOCK_WIND":
        data = pd.read_hdf(h5_local_path + '/ETC/CHINA_STOCK/WIND/ETC_CHINA_STOCK_WIND.h5')
        data.index.name = "stock"
    else:
        if table_name in overwrite_h5:
            source, sub_table_name = wind_str[:-1], table_name[5:]
            h5_path = "{}/{}/{}/{}.h5".format(root_path, source, sub_table_name, sub_table_name)
            end_date = today
        elif wind_str in table_name:
            source, sub_table_name = wind_str[:-1], table_name[5:]
            h5_path = "{}/{}/{}/{}.h5".format(root_path, source, sub_table_name, sub_table_name)
        elif gogoal_str in table_name:
            source, sub_table_name = gogoal_str[:-1], table_name[8:]
            h5_path = "{}/{}/{}/{}.h5".format(root_path, source, sub_table_name, sub_table_name)
        else:
            table_path_list = table_name.split("_")
            table_path = "/".join(table_path_list[:1] + ["_".join(table_path_list[1:3])] + table_path_list[3:])
            h5_path = "/data/group/800080/warehouse/prod/{}/{}.h5".format(table_path, table_name)
        # h5多列提取，形式为"[colA，colB]"
        if col:
            if col[0].startswith("["):
                col = col[0][1: -1].split(",")
                col = list(map(lambda x: x.strip(), col))
            if table_name in mapping_h5:
                #如果子类的depend_data没有筛选字段，新增筛选字段然后rename
                if mapping_h5[table_name] not in col:
                    col.append(mapping_h5[table_name])
                    data = IO.read_data(alt=h5_path, trading_days=[start_date, end_date], columns=col)
                    data = data.rename({mapping_h5[table_name]: "htsc_date"}, axis=1)
                else:
                    # 如果子类的depend_data有筛选字段，新增htsc_date
                    data = IO.read_data(alt=h5_path, trading_days=[start_date, end_date], columns=col)
                    data["htsc_date"] = data[mapping_h5[table_name]]
                #如果stm_issuingdate为筛选字段，需要修改时间格式
                if mapping_h5[table_name] == "stm_issuingdate":
                    data["htsc_date"] = list(map(lambda x: dt.datetime.strptime(x, "%Y-%m-%d"), data["htsc_date"]))
                if mapping_h5[table_name] == "ENTRYDATE":
                    data["htsc_date"] = list(map(lambda x: dt.datetime.strptime(x[:10], "%Y-%m-%d"), data["htsc_date"]))
            else:
                data = IO.read_data(alt=h5_path, trading_days=[start_date, end_date], columns=col)
        else:
            data = IO.read_data(alt=h5_path, trading_days=[start_date, end_date], columns=None)
            if table_name in mapping_h5:
                data["htsc_date"] = data[mapping_h5[table_name]]
                if mapping_h5[table_name] == "stm_issuingdate":
                    data["htsc_date"] = list(map(lambda x: dt.datetime.strptime(x, "%Y-%m-%d"), data["htsc_date"]))
                if mapping_h5[table_name] == "ENTRYDATE":
                    data["htsc_date"] = list(map(lambda x: dt.datetime.strptime(x[:10], "%Y-%m-%d"), data["htsc_date"]))
        data.index.rename(["date", "stock"], inplace=True)
    return data


# 更新h5表的database和datainfo
def __update_single_financial_data(database, data_info, financial_data_type, data, source_str):
    database.update("depend_data", financial_data_type, data)
    data_info["depend_data"].update({financial_data_type: "{}FINANCIAL".format(source_str)})


# 分钟线是按月存的pickle，读取分钟线需要获取所需月份
def __get_month_list(begin_date, last_date):
    begin_month = begin_date[:6]
    last_month = last_date[:6]
    while begin_month <= last_month:
        yield begin_month
        if int(begin_month[-2:]) >= 12:
            begin_month = str(int(begin_month[:4]) + 1) + str("01")
        else:
            begin_month = str(int(begin_month) + 1)


# 分钟数据和日频数据的股票池保持一致
def __process_data_stock_list(data, stock_list):
    data = data.reindex(columns=stock_list)
    return data


def get_single_minute_data(col, month_list, minute_path, begin_date, last_date, stock_list):
    if "-index" in col:
        index_minute_path = minute_local_path + "/index/"
        temp_col = col.split("-")[0]
        res_list = []
        for month in month_list:
            df = pd.read_pickle(
                index_minute_path + "{}/".format(temp_col) + str(month) + "_{}.pkl".format(temp_col))
            res_list.append(df)
        minute_data = pd.concat(res_list, axis=F0)
        minute_data = minute_data.loc[begin_date: last_date]
    else:
        temp_col = col[:-7]
        res_list = []
        for month in month_list:
            df = pd.read_pickle(minute_path + "{}/".format(temp_col) + str(month) + "_{}.pkl".format(temp_col))
            res_list.append(df)
        minute_data = pd.concat(res_list, axis=0)
        minute_data = minute_data.loc[begin_date: last_date]
        minute_data = minute_data.reindex(columns=stock_list)
    return minute_data


# 更新分钟数据的database和datainfo
def __update_minute_data(database, data_info, basic_minute_type_list, begin_date, last_date, basic_str, stock_list):
    # 根据股票组获取分钟线数据时需要指定月份
    month_list = list(__get_month_list(begin_date, last_date))
    minute_path = minute_local_path + "/stock/"
    for col in basic_minute_type_list:
        minute_data = get_single_minute_data(col, month_list, minute_path, begin_date, last_date, stock_list)
        database.update("depend_data", basic_str + col, minute_data)
        data_info["depend_data"].update({basic_str + col: "MINUTE"})

    # 并行取分钟字段
    # with Pool(24) as p:
    #     minute_data_list = p.starmap(get_single_minute_data,
    #                                  zip(basic_minute_type_list, repeat(month_list), repeat(minute_path),
    #                                      repeat(begin_date), repeat(last_date), repeat(stock_list)))
    # for col, minute_data in zip(basic_minute_type_list, minute_data_dict):
    #     database.update("depend_data", basic_str + col, minute_data)
    #     data_info["depend_data"].update({basic_str + col: "MINUTE"})


# 实盘半日分钟数据
def __update_am_minute_data(database, data_info, basic_minute_type_list, basic_str):
    # 获取实盘分钟线数据
    ma = MarketData()
    minute_basic_type_list = list(map(lambda x: x[:-7], basic_minute_type_list))
    data = ma.getAmKline1M4ZTDataFrame(factor_name=minute_basic_type_list)
    for col in minute_basic_type_list:
        data_t = data[col].unstack().astype(np.float64)
        database.update("depend_data", basic_str + col + "_minute", data_t)
        data_info["depend_data"].update({basic_str + col + "_minute": "MINUTE"})


# 获取数据
def __load_data(depend_data_type_list, stock_list, require_date_list, minute_start_date, financial_start_date,
                input_factor_lib, run_type, *args):
    database = Database()
    data_info = {
        "depend_data": {},
        "depend_factors": {},
        "depend_nonfactors": {}
    }

    # 需要判断depend_data_type_list为空的情况（例如业务只用被依赖的因子做rolling等操作）
    if depend_data_type_list:
        # 金工组IO中可能用到所有标准格式h5
        h5str_list = ['INDEXWEIGHT', 'INDUSTRY', 'ETC', 'RISK', 'FDD', 'FCD', 'CALENDAR', 'UNIV', 'VD']
        h5_data_type_list = list(
            filter(lambda x: "_" in x and x.split(".")[1].split("_")[0] in h5str_list, depend_data_type_list))
        # print('h5_data_type_list:\n', h5_data_type_list)
        wind_data_type_list = list(filter(lambda x: wind_str in x, depend_data_type_list))
        # print('wind data type list:\n', wind_data_type_list)
        gogoal_data_type_list = list(filter(lambda x: gogoal_str in x, depend_data_type_list))
        # print('gogoal data type list:\n', gogoal_data_type_list)
        financial_data_type_list = wind_data_type_list + gogoal_data_type_list + h5_data_type_list

        minute_data_type_list = list(
            filter(lambda x: "Basic_factor." in x and "_minute" in x, depend_data_type_list))

        data_type_list = list(filter(lambda x: "Basic_factor." in x and "_minute" not in x, depend_data_type_list))
        daily_data_type_h5_list = ["FactorData.RISK_CHINA_STOCK_DAILY_STYLEFACTOR"]
        daily_data_type_h5_list = list(filter(lambda x: x in daily_data_type_h5_list, depend_data_type_list))
        financial_data_type_list = list(set(financial_data_type_list) - set(daily_data_type_h5_list))

        basic_str = "FactorData.Basic_factor."

        print('data_type_list:\n', data_type_list)

        # 日频基础数据
        if data_type_list:
            # ["close", "volume-000001.SH", ]
            basic_type_list = list(map(lambda x: x.split(".", 2)[2], data_type_list))
            param_basic_type_list = list(filter(lambda x: "-" in x, basic_type_list))
            noparam_basic_type_list = list(filter(lambda x: "-" not in x, basic_type_list))

            date_pkl_list = ["is_universe", "is_valid", "is_valid_raw"]
            date_pkl_list = list(filter(lambda x: x in date_pkl_list, noparam_basic_type_list))

            date_h5_list = ["Data_twap", "Data_suspension", "Data_limit_pctg"]
            date_h5_list = list(filter(lambda x: x in date_h5_list, noparam_basic_type_list))

            by_list = ["volume_by_share", "amt_by_yuan"]

            factordata_basic_type_list = list(
                filter(lambda x: x not in date_pkl_list and x not in date_h5_list and x not in by_list,
                       noparam_basic_type_list))
            print('factordata__basic_type_list:\n', factordata_basic_type_list)

            if factordata_basic_type_list:
                data = fa.get_factor_value('Basic_factor', stock=stock_list, mddate=require_date_list,
                                           factor_names=factordata_basic_type_list, fill_na=True)
                for col in factordata_basic_type_list:
                    data_t = data[col].unstack()
                    data_t = __process_data_stock_list(data_t, stock_list)
                    database.update("depend_data", basic_str + col, data_t)
                    data_info["depend_data"].update({basic_str + col: "DAY"})

            if (set(basic_type_list) & set(by_list)):
                by_data_list = list(map(lambda x: x.split("_by_")[0], by_list))
                data = fa.get_factor_value('Basic_factor', stock=stock_list, mddate=require_date_list,
                                           factor_names=by_data_list, fill_na=True)
                for col, data_name in zip(by_data_list, by_list):
                    data_t = data[col].unstack()
                    if col == "volume":
                        data_t_values = data_t.values * 100
                        data_t = pd.DataFrame(data_t_values, index=data_t.index, columns=data_t.columns)
                    elif col == "amt":
                        data_t_values = data_t.values * 1000
                        data_t = pd.DataFrame(data_t_values, index=data_t.index, columns=data_t.columns)
                    else:
                        pass
                    data_t = __process_data_stock_list(data_t, stock_list)
                    database.update("depend_data", basic_str + data_name, data_t)
                    data_info["depend_data"].update({basic_str + data_name: "DAY"})

            if param_basic_type_list:
                for item in param_basic_type_list:
                    data = fa.get_factor_value("Basic_factor", stock=[item.split("-")[1]], mddate=require_date_list,
                                               factor_names=[item.split("-")[0]], fill_na=True)
                    data_t = data.unstack()
                    database.update("depend_data", basic_str + item, data_t)
                    data_info["depend_data"].update({basic_str + item: "DAY"})
            if date_pkl_list:
                for item in date_pkl_list:
                    data_t = pd.read_pickle(daily_local_path + "/{}.pkl".format(item))
                    data_t = __process_data_stock_list(data_t, stock_list)
                    database.update("depend_data", basic_str + item, data_t)
                    data_info["depend_data"].update({basic_str + item: "DAY"})
            if date_h5_list:
                for item in date_h5_list:
                    data_t = pd.read_hdf(daily_local_path + "/{}.h5".format(item), '/factor')
                    data_t.index = list(map(lambda x: str(x), data_t.index))
                    data_t = __process_data_stock_list(data_t, stock_list)
                    database.update("depend_data", basic_str + item, data_t)
                    data_info["depend_data"].update({basic_str + item: "DAY"})

        if daily_data_type_h5_list:
            table_list = list(map(lambda x: x.split(".")[1], h5_data_type_list))
            col_list = map(lambda x: [x.split(".")[2]] if x.count(".") == 2 else None, h5_data_type_list)
            root_path = h5_local_path + "/DATABASE"
            pre_str = "FactorData."
            for table_name, col in zip(table_list, col_list):
                data = __get_fanancial_data(table_name, require_date_list[0], require_date_list[-1], root_path, col)
                if col:
                    database.update("depend_data", pre_str + table_name + "." + col[0], data)
                    data_info["depend_data"].update({pre_str + table_name + "." + col[0]: "DAY"})
                else:
                    database.update("depend_data", pre_str + table_name, data)
                    data_info["depend_data"].update({pre_str + table_name: "DAY"})

        # 分钟频基础数据
        if minute_data_type_list:
            basic_minute_type_list = list(map(lambda x: x.split(".")[2], minute_data_type_list))
            if run_type == "after":
                begin_date = minute_start_date
                last_date = max(require_date_list)
                __update_minute_data(database, data_info, basic_minute_type_list, begin_date, last_date, basic_str,
                                     stock_list)
            else:
                __update_am_minute_data(database, data_info, basic_minute_type_list, basic_str)

        # 财务原表数据
        if financial_data_type_list:
            end_date = require_date_list[-1]
            start_date = financial_start_date
            root_path = h5_local_path + "/DATABASE"
            pre_str = "FactorData."

            if wind_data_type_list:
                table_list = map(lambda x: x.split(".")[1], wind_data_type_list)
                col_list = map(lambda x: [x.split(".")[2]] if x.count(".") == 2 else None, wind_data_type_list)
                for table_name, col in zip(table_list, col_list):
                    data = __get_fanancial_data(table_name, start_date, end_date, root_path, col)
                    if col:
                        __update_single_financial_data(database, data_info, pre_str + table_name + "." + col[0], data,
                                                       wind_str)
                    else:
                        __update_single_financial_data(database, data_info, pre_str + table_name, data, wind_str)
            if gogoal_data_type_list:
                table_list = list(map(lambda x: x.split(".")[1], gogoal_data_type_list))
                col_list = map(lambda x: [x.split(".")[2]] if x.count(".") == 2 else None, gogoal_data_type_list)
                for table_name, col in zip(table_list, col_list):
                    data = __get_fanancial_data(table_name, start_date, end_date, root_path, col)
                    if col:
                        __update_single_financial_data(database, data_info, pre_str + table_name + "." + col[0], data,
                                                       gogoal_str)
                    else:
                        __update_single_financial_data(database, data_info, pre_str + table_name, data, gogoal_str)

            if h5_data_type_list:
                table_list = list(map(lambda x: x.split(".")[1], h5_data_type_list))
                col_list = map(lambda x: [x.split(".")[2]] if x.count(".") == 2 else None, h5_data_type_list)
                for table_name, col in zip(table_list, col_list):
                    data = __get_fanancial_data(table_name, start_date, end_date, root_path, col)
                    if col:
                        __update_single_financial_data(database, data_info, pre_str + table_name + "." + col[0], data,
                                                       "h5_")
                    else:
                        __update_single_financial_data(database, data_info, pre_str + table_name, data, "h5_")

    if input_factor_lib:
        for depend_name in args[0]:
            data_info[args[1]].update({depend_name: "DAY"})
            data = fa.get_factor_value(input_factor_lib, stock=stock_list, mddate=require_date_list,
                                       factor_names=[depend_name])[depend_name].unstack()
            database.update(args[1], depend_name, data)
    return database, data_info


# 获取指定batch依赖的数据集
def get_database_for_task(task, run_type):
    calc_datetime_list = task["calc_time_list"]
    factor_class_list = task["factor_class_list"]
    stock_list = task["stock_list"]
    full_datetime_list = task["full_datetime_list"]
    input_factor_lib = task["input_factor_lib"]

    max_daily_data_exceed = FactorUtil.get_max_daily_data_exceed(factor_class_list)
    max_minute_data_exceed = FactorUtil.get_max_minute_data_exceed(factor_class_list)
    max_financial_data_exceed = FactorUtil.get_max_financial_data_exceed(factor_class_list)

    start_date, end_date = calc_datetime_list[0], calc_datetime_list[-1]
    start_date_index, end_date_index = full_datetime_list.index(start_date), full_datetime_list.index(end_date)

    require_date_list = full_datetime_list[start_date_index - max_daily_data_exceed: end_date_index + 1]
    minute_start_date = full_datetime_list[start_date_index - max_minute_data_exceed]
    financial_start_date = dt.datetime.strptime(full_datetime_list[start_date_index - max_financial_data_exceed],
                                                "%Y%m%d")

    depend_data_type_list = FactorUtil.get_factor_depend_data_types(factor_class_list)
    depend_nonfactor_list = FactorUtil.get_factor_depend_nonfactors(factor_class_list)
    depend_factor_list = FactorUtil.get_factor_depend_factors(factor_class_list)
    # 通过判断input_factor_lib确定是否需要获取其他depend_factor
    if input_factor_lib == None:
        database, data_info = __load_data(depend_data_type_list, stock_list, require_date_list, minute_start_date,
                                          financial_start_date, input_factor_lib, run_type)
    elif depend_nonfactor_list:
        database, data_info = __load_data(depend_data_type_list, stock_list, require_date_list, minute_start_date,
                                          financial_start_date, input_factor_lib, run_type, depend_nonfactor_list,
                                          "depend_nonfactors")
    else:
        database, data_info = __load_data(depend_data_type_list, stock_list, require_date_list, minute_start_date,
                                          financial_start_date, input_factor_lib, run_type, depend_factor_list,
                                          "depend_factors")
    return database, data_info


# @ray.remote
def update_factor_data_for_ray(factor_name, factor_value, output_factor_lib, start_date, end_date):
    fa = FactorData()
    try:
        fa.add_factor(output_factor_lib, [factor_name])
    except:
        pass
    factor_value.index.name = "mddate"
    factor_value.columns.name = "stock"
    factor_value = factor_value.unstack().to_frame()
    factor_value.columns = [factor_name]
    fa.update_factor_value(output_factor_lib, factor_value, delete_range=(start_date, end_date))


def update_factor_data(factor_name, factor_value, output_factor_lib, start_date, end_date):
    try:
        fa.add_factor(output_factor_lib, [factor_name])
    except:
        pass
    factor_value.index.name = "mddate"
    factor_value.columns.name = "stock"
    factor_value = factor_value.unstack().to_frame()
    factor_value.columns = [factor_name]
    fa.update_factor_value(output_factor_lib, factor_value, delete_range=(start_date, end_date))


def save_factor(result, output_factor_lib, start_date, end_date, num_cpu):
    if num_cpu == 1:
        for factor_name, factor_value in result.items():
            update_factor_data(factor_name, factor_value, output_factor_lib, start_date, end_date)
    else:
        ray_tasks = [
            update_factor_data_for_ray.remote(factor_name, factor_value, output_factor_lib, start_date, end_date)
            for factor_name, factor_value in result.items()]
        ray.get(ray_tasks)
