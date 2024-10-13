# import ray
# from xquant.xqutils.helper import multicore_init
import xfactor.runner.DataManager as DataManager
from xfactor.runner.BasicTaskManager import BasicTaskManager as TaskManager
import datetime as dt
from xfactor.FactorUtil import create_factor_instance
from multiprocessing import Pool
from itertools import repeat


def get_result(factor_name_list, start_date, end_date, calc_num_cpus, fix_config={}, input_factor_lib=None,
               run_type="after", for_temp=None, with_temp=None, fix_times=[]):
    result = {}
    if len(fix_times) > 0:
        for key in factor_name_list:
            fix_config[key] = fix_times
    task_manager = TaskManager(factor_name_list, start_date, end_date, fix_config, input_factor_lib)
    task_list = task_manager.generate_task()
    print("task num:", len(task_list))
    if calc_num_cpus == 1:
        task_results = [__execute_task(task, run_type, for_temp, with_temp) for task in task_list]
    else:
        # with Pool(calc_num_cpus) as p:
        #     task_results = p.starmap(__execute_task,
        #                              zip(task_list, repeat(run_type), repeat(for_temp), repeat(with_temp)))
        raise Exception('I am commented out')
    for task_result in task_results:
        result = __merge_factor_values(result, task_result)
    return result


def run(factor_name_list, start_date, end_date, fix_config={}, input_factor_lib=None, output_factor_lib=None,
        save=False, options=None, run_type="after", fix_times=[]):
    calc_num_cpus = 1
    if options is not None and "calc.num_cpus" in options:
        calc_num_cpus = int(options["calc.num_cpus"])
    result = get_result(factor_name_list, start_date, end_date, calc_num_cpus, fix_config, input_factor_lib, run_type,
                        None, None, fix_times)
    # if save:
    #     t0 = dt.datetime.now()
    #     save_num_cpu = 1
    #     if options is not None and "update.num_cpus" in options:
    #         save_num_cpu = int(options["update.num_cpus"])
    #         assert multicore_init() == True
    #         ray.init(num_cpus=save_num_cpu)
    #         DataManager.save_factor(result, output_factor_lib, start_date, end_date, save_num_cpu)
    #         ray.shutdown()
    #     else:
    #         DataManager.save_factor(result, output_factor_lib, start_date, end_date, save_num_cpu)
    #     print("update factor cost: ", dt.datetime.now() - t0)
    #     return result
    # else:
    #     return result
    return result


def run_for_temp(factor_name_list, start_date, end_date, fix_config={}, input_factor_lib=None, run_type="after",
                 temp_path=None, fix_times=[]):
    temp_result = get_result(factor_name_list, start_date, end_date, fix_config, input_factor_lib, run_type, temp_path,
                             None, fix_times)
    if temp_path and temp_result:
        for key, value in temp_result.items():
            value.to_pickle("{}/{}.pkl".format(temp_path, key))
        print("save temp result done")


def run_with_temp(factor_name_list, start_date, end_date, fix_config={}, input_factor_lib=None, output_factor_lib=None,
                  save=False, options=None, run_type="after", temp_path=None, fix_times=[]):
    result = get_result(factor_name_list, start_date, end_date, fix_config, input_factor_lib, run_type, None, temp_path,
                        fix_times)
    if save:
        t0 = dt.datetime.now()
        save_num_cpu = 1
        if options is not None and "update.num_cpus" in options:
            save_num_cpu = int(options["update.num_cpus"])
        assert multicore_init() == True
        ray.init(num_cpus=save_num_cpu)
        DataManager.save_factor(result, output_factor_lib, start_date, end_date, save_num_cpu)
        ray.shutdown()
        print("update factor cost: ", dt.datetime.now() - t0)
        return result
    else:
        return result

def __multi_execute_task(task, run_type, for_temp=None, with_temp=None):
    result = {}
    database, data_info = DataManager.get_database_for_task(task, run_type)
    for factor_class in task["factor_class_list"]:
        factor_instance = create_factor_instance(factor_class, task["fix_config"])
        calc_result = factor_instance.calc(database, data_info, task["calc_time_list"], task["full_datetime_list"],
                                           for_temp, with_temp)
        result = __merge_factor_values(result, calc_result)
    return result

# 运行指定task
def __execute_task(task, run_type, for_temp=None, with_temp=None):
    result = {}
    time1 = dt.datetime.now()
    database, data_info = DataManager.get_database_for_task(task, run_type)
    time2 = dt.datetime.now()
    print("Load data cost:", time2 - time1)
    for factor_class in task["factor_class_list"]:
        print('creating instance for factor_class: {}, type: {}'.format(factor_class, type(factor_class)))
        factor_instance = create_factor_instance(factor_class, task["fix_config"])
        calc_result = factor_instance.calc(database, data_info, task["calc_time_list"], task["full_datetime_list"],
                                           for_temp, with_temp)
        result = __merge_factor_values(result, calc_result)
    time3 = dt.datetime.now()
    print("Calculate factor cost:", time3 - time2)
    return result


def __merge_factor_values(result, calc_result):
    for factor_name in calc_result:
        if factor_name in result:
            result[factor_name] = result[factor_name].append(calc_result[factor_name])
        else:
            result[factor_name] = calc_result[factor_name]
    return result
