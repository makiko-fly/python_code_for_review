from xfactor.runner.BasicTaskManager import BasicTaskManager


# 不针对时间维度进行分割，将每个因子分割
class TaskManager(BasicTaskManager):
    max_factor_num_per_task = 9999
    max_date_num_per_task = 1
