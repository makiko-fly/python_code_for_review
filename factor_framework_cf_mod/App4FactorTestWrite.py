import xfactor.runner.Runner as Runner
# import xfactor.runner.BasicRunner as BasicRunner
import xfactor.test.TesterWithRay as Tester
from xfactor.FactorUtil import get_factor_class
import pandas as pd
import json
import ray
import shutil
from xquant.factordata import FactorData

s = FactorData()
# factor_list = [i[:-3] for i in os.listdir('./factor_without_test/') if i[-3:] == '.py']
factor_list = ['RankEBITPSChg']
factor_qualified = []

for fac in factor_list:
    x_day_lib = s.get_library_info()['x_day_lib']
    factor_type = get_factor_class(fac).factor_type
    if factor_type == 'DAY':
        x_day_lib = [i for i in x_day_lib if i[:3] != 'Fix']
    elif factor_type == 'FIX':
        x_day_lib = set([i for i in x_day_lib if i[:3] == 'Fix'])
    assert fac not in x_day_lib, '{} has been in x_day_lib already! Please change the name!'.format(fac)
    try:
        res = Runner.run([fac], 20140101, 20190630,
                         options={'ray.num_cpus': 24, 'ray.object_store_memory': 10 ** 9 * 150})
        if factor_type == 'DAY':
            test_result = Tester.test(20160101, 20190630, res,
                                      [20160630, 20161231, 20170630, 20171231, 20180630, 20181231],
                                      local=False)
            if test_result:
                res = Runner.run([fac], 20140101, 20200110, save=True, output_factor_lib='x_day_lib',
                                 options={'ray.num_cpus': 24, 'ray.object_store_memory': 10 ** 9 * 150})
                # factor_names = list(res.keys())
                Tester.merge_test_data(res)
                factor_qualified.append(fac)
        elif factor_type == 'FIX':
            test_result, result_dict = Tester.test(20160101, 20190630, res,
                                                   [20160630, 20161231, 20170630, 20171231, 20180630, 20181231],
                                                   local=False)
            if test_result:
                result_sr = pd.Series(result_dict)
                true_fix = result_sr[result_sr].index.tolist()
                with open('config.json', 'r') as f:
                    fix_config = json.load(f)
                fix_config[fac] = true_fix
                with open('config.json', 'w') as f:
                    json.dump(fix_config, f)
                res = Runner.run([fac], 20140101, 20190630, save=True, output_factor_lib='x_day_lib',
                                 fix_config=fix_config, options={
                        'ray.num_cpus': 24, 'ray.object_store_memory': 10 ** 9 * 150, 'update_num_cpu': 7})
                # factor_names = list(res.keys())
                Tester.merge_test_data(res)
                factor_qualified.append(fac)
    except Exception as e:
        ray.shutdown()
        print(e)

for fac in factor_qualified:
    shutil.copy('./factor_without_test/{}.py'.format(fac), './factor/[}.py'.format(fac))
