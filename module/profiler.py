from nasbench import api
import numpy as np

import sys, os

class Profiler:

    def __init__(self, benchmark_name):

        self.benchmark_name = benchmark_name

        if benchmark_name.startswith('nasbench101') :
                if benchmark_name == 'nasbench101_full':
                    self.bench = api.NASBench("/home/zengxia6/research/NAS/nasbench_data/nasbench_full.tfrecord")
                else :
                    self.bench = api.NASBench("/home/zengxia6/research/NAS/nasbench_data/nasbench_only108.tfrecord")



    def profile(self, model, budget=None):

        if self.benchmark_name.startswith('nasbench101'):
            if not budget:
                budget = 108
            fixed_stat, computed_stat = self.bench.get_metrics_from_spec(model.arch.modelspec())


            i = np.random.randint(3)
            val_acc = computed_stat[budget][i]["final_validation_accuracy"]


            test_acc = np.mean([computed_stat[108][i]["final_test_accuracy"] for i in range(3)])
            train_time = np.mean([computed_stat[budget][i]["final_training_time"] for i in range(3)])
            params = fixed_stat['trainable_parameters']
            cost = train_time * 108 / budget

            return {'validation_accuracy':val_acc, 'training_time':train_time,
                    'trainable_parameters':params/ 5e7, 'test_accuracy':test_acc, 'cost':cost / 5e3}



    def is_valid(self, arch):

        if self.benchmark_name.startswith('nasbench101'):
            return self.bench.is_valid(api.ModelSpec(**arch))

