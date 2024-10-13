# import xfactor.runner.Runner as Runner
import xfactor.runner.BasicRunner as Runner
import time
import sys
import os
import json

# add customized factor class dir to sys.path
from limit_config import limit_config
if 'factor_classes_dir' in limit_config:
    factor_cls_dir = limit_config['factor_classes_dir'].strip()
    if factor_cls_dir == '':
        pass
    else:
        assert os.path.exists(factor_cls_dir), '{} in limit config is not a valid dir'.format(factor_cls_dir)
        sys.path.insert(0, factor_cls_dir)

factor_name_list = ["MinVwapRV_mod"]

start = time.time()
#可以在config里面修改因子类的fix_times属性，可以不填，默认fix_times为7个。如果在子类里面定义了该属性，那么以子类里面为准。
with open("config.json") as f:
    fix_config = json.load(f)

#factor_name_list： 需要计算的因子列表, 必需
#save：表示是否将因子存入因子库，默认为False，非必需；
#output_factor_lib：存入的因子库名，非必需；
#fix_config: 可以在json里面修改fix_times，方便后期更新fix因子，非必需；
#options：ray.num_cpus表示因子计算的并行数，默认为4；update.num_cpu表示因子入库的并行数，默认为4。非必需
res = Runner.run(factor_name_list, 20181201, 20190101, save=False, output_factor_lib="zero_alpha", fix_config=fix_config,
                 options={"ray.num_cpus": 8, "update.num_cpu": 8})
# print(res)
print("total cost time:", time.time() - start)
