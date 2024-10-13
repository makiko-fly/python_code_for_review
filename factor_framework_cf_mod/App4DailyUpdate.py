import xfactor.runner.Runner as Runner
import time
import datetime as dt
import json

# 因子值入库计算
start = time.time()
# 自己定义
factor_name_list = ["SampleFactor", "GTJA001", "VNSPMean_1_5"]
today = dt.datetime.strftime(dt.datetime.today(), "%Y%m%d")

# 如果fix因子通过检测，需要配置fix_config作为传入参数。且在fix因子子类中不能重写fix_time参数
with open("config.json") as f:
    fix_config = json.load(f)

res = Runner.run(factor_name_list, today, today, output_factor_lib="x_day_lib", fix_config=fix_config, save=True,
                 options={"ray.num_cpus": 8})
