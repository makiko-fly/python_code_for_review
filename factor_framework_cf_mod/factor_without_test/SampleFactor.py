from xfactor.BaseFactor import BaseFactor
from xfactor.FixUtil import minute_data_transform

class SampleFactor(BaseFactor):
    #  定义因子参数

    # 因子频率，默认为日频因子，可不设置
    factor_type = "FIX"
    fix_times = ["1300"]
    # 依赖的平台原始数据，包括FactorData和MarketData接口中的数据。 默认为空，必须设置
    # depend_data = [ "FactorData.Basic_factor.pct_chg", "FactorData.Basic_factor.volume_minute", "FactorData.WIND_AShareProfitExpress"]
    depend_data = [ "FactorData.Basic_factor.pct_chg", "FactorData.Basic_factor.volume_minute"]
    # 计算每个时点的因子所需要前移的数据窗口大小
    # 例如，为日频因子，lag=3表示计算某一日的因子值需要依赖前三个交易日和当日的数据，默认为0，可不设置
    lag = 1
    financial_lag = 200
    # 播放后得到的结果，可按照该长度进行rolling等计算，具体rolling方法需要在reform方法中定义。 默认为1，可不设置。
    # reform_window = 10

    # 每次播放的计算具体方法。必须实现。
    def calc_single(self, database):
        minute_data_transform(database.depend_data, ["drop", "merge"])
        #读取分钟数据
        volume_minute = database.depend_data['FactorData.Basic_factor.volume_minute']
        #读取日频数据
        pct_chg = database.depend_data['FactorData.Basic_factor.pct_chg']
        #读取财务数据，暂时使用h5表数据
        # close = database.depend_data['FactorData.WIND_AShareProfitExpress']
        ans = pct_chg.iloc[-1, :]
        ans.fillna(0, inplace=True)
        return ans

    # # 针对播放后的结果，进行相关的rolling等操作。所用的前序数据长度应为reform_window。默认不修改temp_result， 可不重写。
    # def reform(self, temp_result):
    #     return temp_result.rolling(self.reform_window).mean()
