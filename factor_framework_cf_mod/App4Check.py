import xfactor.runner.Runner as Runner
from xquant.factordata import FactorData

fa = FactorData()


def check_factor(start_date, end_date, res):
    lib_factors = fa.get_factor_value("x_day_lib", mddate=fa.tradingday(start_date, end_date),
                                      factor_names=list(res.keys()))
    wrong_factors = []
    for factor_name, factor_value in res.items():
        lib_factor_value = lib_factors[factor_name].unstack()
        cols = list(set(lib_factor_value.columns) & set(factor_value.columns))
        lib_factor_value = lib_factor_value[cols].astype(float)
        factor_value = factor_value[cols].astype(float)
        if not lib_factor_value.equals(factor_value):
            wrong_factors.append(factor_name)
    return wrong_factors


def check_factor_by_lag(end_date, res, lag=5):
    start_date = fa.tradingday(end_date, -lag)
    return check_factor(start_date, end_date, res)

# 筛查因子的类名列表，需在因子库中存在。
factor_name_list = ["AmtKurt"]
# 筛查区段，可以自己设置，需要该时间段内有因子值
start_date = 20200410
end_date = 20200410

res = Runner.run(factor_name_list, start_date, end_date, save=False, options={"ray.num_cpus": 8})
# 输出不一致因子列表
wrong_factors = check_factor(start_date, end_date, res)
print(wrong_factors)
