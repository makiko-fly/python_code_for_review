import numpy as np
import pandas as pd
from xfactor.BaseFactor import BaseFactor
from xfactor.FixUtil import minute_data_transform

'''
* 因子名：MinVwapRV
* 逻辑：该因子是一个分钟因子，首先以 涨跌幅/成交量 这一标准筛选出较为有效的交易时间段，再比较vwap。
*      这实际上反映了具有信息优势的交易者的高抛低吸的行为，vwap比值越低说明这些交易者在逢低吸筹，比值较高
*      则说明在高位抛货，跟随他们的行为可以获利
* 因子参数：分钟数据的高开低收
* 日期：2019.1.24
* 函数修改日期：尚未修改
* 修改人：尚未修改
* 修改原因：尚未修改
* 迁移日期：2020.1.10
'''


class MinVwapRV(BaseFactor):
    factor_type = "DAY"
    fix_times = ["1500"]

    # depend_data = ["FactorData.Basic_factor.open_adj_minute",
    #                "FactorData.Basic_factor.close_adj_minute",
    #                "FactorData.Basic_factor.volume_minute",
    #                "FactorData.Basic_factor.amt_minute"]

    depend_data = ["FactorData.Basic_factor.volume_minute",
                   "FactorData.Basic_factor.amt_minute"]

    lag = 0
    # reform_window = 20  # 降换手

    def calc_single(self, single_database):
        minute_data_transform(single_database.depend_data, operation=["drop", "merge"])

        # open_minute = single_database.depend_data["FactorData.Basic_factor.open_adj_minute"]
        # close_minute = single_database.depend_data["FactorData.Basic_factor.close_adj_minute"]
        volume_minute = single_database.depend_data["FactorData.Basic_factor.volume_minute"]
        amt_minute = single_database.depend_data["FactorData.Basic_factor.amt_minute"]
        #
        # ret_minute = pd.DataFrame(close_minute.values / open_minute.values - 1, index=close_minute.index,
        #                           columns=close_minute.columns)
        # RV_minute = pd.DataFrame(np.abs(ret_minute.values) / volume_minute.values, index=close_minute.index,
        #                          columns=close_minute.columns)

        # RV_flag = pd.DataFrame(RV_minute.values > RV_minute.quantile(0.8).values,
        #                        index=RV_minute.index,
        #                        columns=RV_minute.columns)

        # vwap_RV = amt_minute[RV_flag].sum().values / volume_minute[RV_flag].sum().values
        #
        # vwap_allday = amt_minute.sum().values / volume_minute.sum().values

        # arr = 1. * vwap_RV / vwap_allday

        ans = pd.Series(1, index=volume_minute.columns)

        return ans

    def reform(self, temp_result):
        alpha = temp_result.rolling(window=20, min_periods=10).mean()
        return alpha