from xquant.factordata import FactorData

fa = FactorData()


def get_factor_names(factor_type):
    factor_list = fa.get_library_info()["x_day_lib"]
    if factor_type == "FIX":
        return list(filter(lambda x: x[:3] == "Fix", factor_list))
    elif factor_type == "DAY":
        return list(filter(lambda x: x[:3] != "Fix", factor_list))
    else:
        raise Exception
