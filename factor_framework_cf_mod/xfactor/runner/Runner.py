import ray
from xquant.xqutils.helper import multicore_init
import xfactor.runner.DataManager as DataManager
from xfactor.runner.DailyUpdateTaskManager import TaskManager as TaskManager
from xfactor.runner.BasicTaskManager import BasicTaskManager as BasicTaskManager
from xquant.factordata import FactorData
import datetime as dt
import itertools
from xfactor.FactorUtil import create_factor_instance

fa = FactorData()


def ray_get_result(factor_name_list, start_date, end_date, fix_config={}, input_factor_lib=None,
                   run_type="after", for_temp=None, with_temp=None, fix_times=[]):
    result = {}
    time1 = dt.datetime.now()
    if len(fix_times) > 0:
        for key in factor_name_list:
            fix_config[key] = fix_times
    if len(factor_name_list) <= 3:
        task_manager = BasicTaskManager(factor_name_list, start_date, end_date, fix_config, input_factor_lib)
    else:
        task_manager = TaskManager(factor_name_list, start_date, end_date, fix_config, input_factor_lib)

    task_list = task_manager.generate_task()
    id_lists = ray.get([__execute_task.remote(task, run_type, for_temp, with_temp) for task in task_list])
    undo_ids = list(itertools.chain(*id_lists))
    time2 = dt.datetime.now()
    print("Total tiny task num:", len(undo_ids))
    print("prepare time cost:", time2 - time1)
    while len(undo_ids):
        done_ids, undo_ids = ray.wait(undo_ids, min(200, len(undo_ids)))
        sub_results = ray.get(done_ids)
        result = __merge_factor_values(result, sub_results)

    time3 = dt.datetime.now()
    print("calc time cost:", time3 - time2)
    return result


# 用于每日更新因子
def run(factor_name_list, start_date, end_date, fix_config={}, input_factor_lib=None, output_factor_lib=None,
        save=False, options=None, run_type="after", fix_times=[]):
    num_cpus = 4
    object_store_memory = 10 ** 9 * 20
    if options is not None:
        if "ray.redis_address" in options:
            redis_address = options["ray.redis_address"]
            assert multicore_init() == True
            ray.init(redis_address=redis_address)
            for node in ray.nodes():
                print("Ray cluster info: ", node["NodeManagerAddress"], node["Resources"], node["alive"])
        else:
            if "ray.num_cpus" in options:
                num_cpus = int(options["ray.num_cpus"])
            if "ray.object_store_memory" in options:
                object_store_memory = options["ray.object_store_memory"]
            assert multicore_init() == True
            ray.init(num_cpus=num_cpus, object_store_memory=object_store_memory)
    else:
        assert multicore_init() == True
        ray.init()

    result = ray_get_result(factor_name_list, start_date, end_date, fix_config, input_factor_lib, run_type, None, None,
                            fix_times)
    if save:
        DataManager.save_factor(result, output_factor_lib, start_date, end_date, num_cpus)
    ray.shutdown()
    return result


def run_for_temp(factor_name_list, start_date, end_date, fix_config={}, input_factor_lib=None, options=None,
                 run_type="after", temp_path=None, fix_times=[]):
    num_cpus = 4
    object_store_memory = 10 ** 9 * 20
    if options is not None:
        if "ray.redis_address" in options:
            redis_address = options["ray.redis_address"]
            assert multicore_init() == True
            ray.init(redis_address=redis_address)
        else:
            if "ray.num_cpus" in options:
                num_cpus = int(options["ray.num_cpus"])
            if "ray.object_store_memory" in options:
                object_store_memory = options["ray.object_store_memory"]
            assert multicore_init() == True
            ray.init(num_cpus=num_cpus, object_store_memory=object_store_memory)
    else:
        assert multicore_init() == True
        ray.init()
    temp_result = ray_get_result(factor_name_list, start_date, end_date, fix_config, input_factor_lib, run_type,
                                 temp_path, None, fix_times)
    ray.shutdown()
    if temp_path and temp_result:
        for key, value in temp_result.items():
            value.to_pickle("{}/{}.pkl".format(temp_path, key))
        print("save temp result done")


def run_with_temp(factor_name_list, start_date, end_date, fix_config={}, input_factor_lib=None, output_factor_lib=None,
                  save=False, options=None, run_type="after", temp_path=None, fix_times=[]):
    num_cpus = 4
    object_store_memory = 10 ** 9 * 20
    if options is not None:
        if "ray.redis_address" in options:
            redis_address = options["ray.redis_address"]
            assert multicore_init() == True
            ray.init(redis_address=redis_address)
        else:
            if "ray.num_cpus" in options:
                num_cpus = int(options["ray.num_cpus"])
            if "ray.object_store_memory" in options:
                object_store_memory = options["ray.object_store_memory"]
            assert multicore_init() == True
            ray.init(num_cpus=num_cpus, object_store_memory=object_store_memory)
    else:
        assert multicore_init() == True
        ray.init()
    result = ray_get_result(factor_name_list, start_date, end_date, fix_config, input_factor_lib, run_type, None,
                            temp_path, fix_times)
    if save:
        DataManager.save_factor(result, output_factor_lib, start_date, end_date, num_cpus)
    ray.shutdown()
    return result


# 运行指定task
@ray.remote
def __execute_task(task, run_type, for_temp=None, with_temp=None):
    database, data_info = DataManager.get_database_for_task(task, run_type)
    database_id = ray.put(database)
    data_info_id = ray.put(data_info)
    calc_time_list_id = ray.put(task["calc_time_list"])
    full_time_list_id = ray.put(task["full_datetime_list"])
    fix_config = ray.put(task["fix_config"])

    return [
        __execute_calculator.remote(factor_class, fix_config, database_id, data_info_id, calc_time_list_id,
                                    full_time_list_id, for_temp, with_temp)
        for factor_class in task["factor_class_list"]]


@ray.remote
def __execute_calculator(factor_class, fix_config, database, data_info, calc_time_list, full_datetime_list, for_temp,
                         with_temp):
    calculator_instance = create_factor_instance(factor_class, fix_config)
    return calculator_instance.calc(database, data_info, calc_time_list, full_datetime_list, for_temp, with_temp)


def __merge_factor_values(result, sub_results):
    for sub_result in sub_results:
        for factor_name in sub_result:
            if factor_name in result:
                result[factor_name] = result[factor_name].append(sub_result[factor_name])
            else:
                result[factor_name] = sub_result[factor_name]
    return result
