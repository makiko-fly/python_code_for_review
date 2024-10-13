import xfactor.runner.BasicRunner as Runner
import xfactor.test.TesterWithRay as Tester
import time

# 本地检测模式
# 计算因子值，TestDay
factor_name = "TestDay"
start = time.time()
# # 计算时间开始时间前移一段时间
res = Runner.run([factor_name], 20140101, 20181231, options={"ray.num_cpus": 8})
test_result = Tester.test(20160101, 20181231, res, [20160630, 20161231, 20170630, 20171231, 20180630, 20181231])
print(test_result)
print("total cost time:", time.time() - start)
