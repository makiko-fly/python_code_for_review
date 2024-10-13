import importlib


def get_factor_depend_data_types(factor_class_list):
    depend_data_types = set()
    for factor_class in factor_class_list:
        depend_data_types.update(factor_class.depend_data)
    return list(depend_data_types)


def get_factor_depend_factors(factor_class_list):
    depend_factors = set()
    for factor_class in factor_class_list:
        depend_factors.update(factor_class.depend_factors)
    return list(depend_factors)


def get_factor_depend_nonfactors(factor_class_list):
    depend_nonfactors = set()
    for factor_class in factor_class_list:
        depend_nonfactors.update(factor_class.depend_nonfactors)
    return list(depend_nonfactors)


def get_todo_factor_list(factor_list, exist_data_types=[], add_data_types=[]):
    new_data_types = exist_data_types + add_data_types
    factor_class_list = get_factor_class_list(factor_list)
    data_type_factor_dict = grouping_factor_by_data_types(factor_class_list)
    exist_factor_set = factors_by_data_types(exist_data_types, data_type_factor_dict)
    todo_factor_set = factors_by_data_types(new_data_types, data_type_factor_dict) - exist_factor_set
    return list(todo_factor_set)


def factors_by_data_types(exist_data_types, data_type_factor_dict):
    unexist_data_types = data_type_factor_dict.keys() - set(exist_data_types)
    factor_set = set([])
    for i in exist_data_types:
        factor_set = factor_set | set(data_type_factor_dict[i])
    for i in unexist_data_types:
        factor_set = factor_set - set(data_type_factor_dict[i])
    return factor_set


def grouping_factor_by_data_types(factor_class_list):
    datatype_factor_dict = {"Day": [], "Minute": [], "Financial": []}
    for factor_class in factor_class_list:
        factor_name = factor_class.get_factor_class_name()
        data_types = factor_class.depend_data
        daily_data_types = list(filter(lambda x: "Basic_factor" in x and x[-7:] != "_minute", data_types))
        if daily_data_types:
            datatype_factor_dict["Day"].append(factor_name)
        minute_data_types = list(filter(lambda x: "Basic_factor" in x and x[-7:] == "_minute", data_types))
        if minute_data_types:
            datatype_factor_dict["Minute"].append(factor_name)
        h5_data_types = list(filter(lambda x: x not in daily_data_types and x not in minute_data_types, data_types))
        if h5_data_types:
            datatype_factor_dict["Financial"].append(factor_name)
    return datatype_factor_dict


def get_max_data_exceed(factor_class_list):
    daily_data_exceed = get_max_daily_data_exceed(factor_class_list)
    minute_data_exceed = get_max_minute_data_exceed(factor_class_list)
    financial_data_exceed = get_max_financial_data_exceed(factor_class_list)
    print('daily_data_exceed: {}, minute_data_exceed: {}, financial_data_exceed: {}'.
          format(daily_data_exceed, minute_data_exceed, financial_data_exceed))
    return max(daily_data_exceed, minute_data_exceed, financial_data_exceed)


def get_max_daily_data_exceed(factor_class_list):
    return max([factor_class.lag + max(factor_class.reform_window - 1, 0) for factor_class in factor_class_list])


def get_max_minute_data_exceed(factor_class_list):
    minute_lag_list = []
    for factor_class in factor_class_list:
        if factor_class.minute_lag is None:
            minute_lag_list.append(factor_class.lag + max(factor_class.reform_window - 1, 0))
        else:
            minute_lag_list.append(factor_class.minute_lag + max(factor_class.reform_window - 1, 0))
    return max(minute_lag_list)


def get_max_financial_data_exceed(factor_class_list):
    return max(
        [factor_class.financial_lag + max(factor_class.reform_window - 1, 0) for factor_class in factor_class_list])


def get_class(kls):
    parts = kls.split(".")
    module = ".".join(parts[:-1])
    if check_module(module):
        m = __import__(module)
        for comp in parts[1:]:
            m = getattr(m, comp)
        return m
    else:
        raise Exception("No factor module:" + module)


def check_module(module_name):
    """检查模块时候能被导入而不用实际的导入模块"""
    return importlib.util.find_spec(module_name)


def get_factor_module(factor_name):
    if check_module(".".join(["factor_classes", factor_name])):
        return "factor_classes"
    elif check_module(".".join(["factor", factor_name])):
        return "factor"
    elif check_module(".".join(["nonfactor", factor_name])):
        return "nonfactor"
    elif check_module(".".join(["factor_without_test", factor_name])):
        return "factor_without_test"
    elif check_module(".".join(["nonfactor_without_test", factor_name])):
        return "nonfactor_without_test"
    else:
        return None


def get_factor_class(factor_name):
    module_name = get_factor_module(factor_name)
    if module_name:
        return get_class(".".join([module_name, factor_name, factor_name]))
    else:
        raise Exception("No factor found:" + factor_name)


def get_factor_class_list(factor_name_list):
    return [get_factor_class(factor_name) for factor_name in factor_name_list]


def is_same_freq_factor(factor_class_list):
    freq_set = set()
    for factor in factor_class_list:
        freq_set.add(factor.factor_type)
    return len(freq_set) == 1


def create_factor_instance(factor_class, fix_config):
    factor_class_name = factor_class.get_factor_class_name()
    if factor_class_name in fix_config:
        args = {"fix_times": fix_config[factor_class_name]}
        factor_instance = factor_class(**args)
    else:
        factor_instance = factor_class()
    return factor_instance