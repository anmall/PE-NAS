import os
import argparse
import collections
import random
import json
import time
import numpy as np
from scipy import stats
import heapq
from multiprocessing import Process
from module.model import Model, ModelHelper
from module.predictor import DataPreparer, Predictor
from module.tracker import Tracker
from module.profiler import Profiler
import joblib


class AsyncScheduler():

    def __init__(self, profiler,  n_workers=8 ):
        self.timers = []
        self.n_workers = n_workers
        self.profiler = profiler

        assert self.n_workers >= 1

        self.curTime = 0


    def parallel_train_and_evaluate(self, model):

        assert len(self.timers) < self.n_workers, "Workers are all busy!"

        result = self.profiler.profile(model)
        model.accuracy = result['validation_accuracy']
        model.training_time = result['training_time']
        model.cost = result['cost']
        model.params = result['trainable_parameters']
        model.test_accuracy = result['test_accuracy']

        heapq.heappush(self.timers, (self.curTime + model.training_time, model))  # (finish time, model)



    def fetch_next_evaluating_model(self):

        if len(self.timers) > 0:
            next_available_time, evaluated_model = heapq.heappop(self.timers)
            ## simulate time ellapsed
            self.curTime = next_available_time
            evaluated_model.finished_time = self.curTime
            return evaluated_model

        return None

    def wait_for_finish(self):

        finished_models = []

        while len(self.timers) > 0:
            next_available_time, evaluated_model = heapq.heappop(self.timers)
            self.curTime = next_available_time
            evaluated_model.finished_time = self.curTime
            finished_models.append(evaluated_model)

        return finished_models

    def full(self):
        return len(self.timers) == self.n_workers


def parallel_regularized_evolution(profiler, args):
    population = collections.deque()
    # kill_over_time = dict()
    scheduler = AsyncScheduler(profiler, n_workers=args.n_workers)
    tracker = Tracker()
    skipped_archs = 0
    non_skipped_archs = 0
    if args.predict_model:
        dp = DataPreparer(args.benchmark)
        predictor = Predictor()


    step = 0

    # Initialize the population with random models.
    init_size = args.sample_size
    while step < init_size:
        model = Model()
        model.arch = ModelHelper.random_cell(profiler)
        while scheduler.full():
            m = scheduler.fetch_next_evaluating_model()
            tracker.record_model(m)
            population.append(m)


        scheduler.parallel_train_and_evaluate(model)
        step += 1

    finished_models = scheduler.wait_for_finish()
    tracker.record_model(finished_models)
    population += finished_models

    if args.predict_model:
        dp.load_evaluated_models(tracker.evaluated_models)
        predictor.fit(dp)
        cached_models = []



    while step < args.n_iters:

        # print(step, 'Pop ({}) Accuracies'.format(len(population)), [model.accuracy for model in population][:10])

        if scheduler.full():

            evaluated_model = scheduler.fetch_next_evaluating_model()
            tracker.record_model(evaluated_model)



            #update predictor
            if args.predict_model:
                if not evaluated_model.es:
                    cached_models.append(evaluated_model)
                if len(cached_models) == 10:
                    dp.load_evaluated_models(cached_models)
                    cached_models = []
                    predictor.fit(dp)

            ## update population
            if not evaluated_model.es:
                population.append(evaluated_model)


            while len(population) > args.pop_size:
                if args.predict_kill:
                    # Sample randomly chosen models from the current population.
                    sample = []
                    while len(sample) < args.sample_size:
                        # Inefficient, but written this way for clarity. In the case of neural
                        # nets, the efficiency of this line is irrelevant because training neural
                        # nets is the rate-determining step.
                        candidate = random.choice(list(population))
                        sample.append(candidate)
                    predicted_accuracies = []
                    for each in sample:
                        candidates_acc = [predictor.predict(np.array([dp.json_model_feature(each.serialize())]))[0]]
                        # candidates_acc = [each.accuracy]
                        while True:
                            c = Model()

                            c.arch = ModelHelper.mutate(each, args.mutate_step, profiler)
                            result = profiler.profile(c, return_fixed_metrics=True)
                            c.training_time = result['training_time']
                            c.cost = result['cost']
                            c.params = result['trainable_parameters']

                            accuracy = predictor.predict(np.array([dp.json_model_feature(c.serialize())]))[0]
                            candidates_acc.append(accuracy)

                            if len(candidates_acc) == args.kill_num:
                                break
                        arch_predicted_acc = np.mean(candidates_acc) + args.kill_alpha * np.std(candidates_acc)
                        predicted_accuracies.append(arch_predicted_acc)
                    min_id = np.argmin(predicted_accuracies)
                    # if step % 20 == 0:
                    #     kill_over_time[step] = ([sample[min_id].accuracy], [sample[min_id].test_accuracy])
                    population.remove(sample[min_id])
                else:
                    if args.method == 're':
                        population.popleft()



        while not scheduler.full():  # fill them

            # Sample randomly chosen models from the current population.
            sample = []
            while len(sample) < args.sample_size:
                # Inefficient, but written this way for clarity. In the case of neural
                # nets, the efficiency of this line is irrelevant because training neural
                # nets is the rate-determining step.
                candidate = random.choice(list(population))
                sample.append(candidate)

            predicted_accuracies = []
            for each in sample:
                candidates_acc = [predictor.predict(np.array([dp.json_model_feature(each.serialize())]))[0]]
                # candidates_acc = [each.accuracy]
                while True:
                    c = Model()

                    c.arch = ModelHelper.mutate(each, args.mutate_step, profiler)
                    result = profiler.profile(c, return_fixed_metrics=True)
                    c.training_time = result['training_time']
                    c.cost = result['cost']
                    c.params = result['trainable_parameters']

                    accuracy = predictor.predict(np.array([dp.json_model_feature(c.serialize())]))[0]
                    candidates_acc.append(accuracy)

                    if len(candidates_acc) == args.kill_num:
                        break
                arch_predicted_acc = np.mean(candidates_acc) + args.kill_alpha * np.std(candidates_acc)
                predicted_accuracies.append(arch_predicted_acc)
            max_id = np.argmax(predicted_accuracies)
            parent = sample[max_id]


            if args.predict_mutate:

                candidate_children = []
                while True:
                    c = Model()

                    c.arch = ModelHelper.mutate(parent, args.mutate_step, profiler)
                    result = profiler.profile(c, return_fixed_metrics=True)
                    c.training_time = result['training_time']
                    c.cost = result['cost']
                    c.params = result['trainable_parameters']


                    c.accuracy = predictor.predict(np.array([dp.json_model_feature(c.serialize())]))[0]
                    candidate_children.append(c)

                    if len(candidate_children) == args.mutate_num:
                        break

                child = max(candidate_children, key=lambda i: i.fitness(mo=args.mo))

            else:
                child = Model()
                child.arch = ModelHelper.mutate(parent, args.mutate_step, profiler)

            child.parent = parent
            child.root = parent.root if parent.root else parent
            scheduler.parallel_train_and_evaluate(child)
            step += 1


            if step % 1000 == 0:

                print('step', step, 'population', stats.describe([m.accuracy for m in population]))




    finished_models = scheduler.wait_for_finish()
    tracker.record_model(finished_models)

    return tracker


def run_trial(i, profiler, args):
    s = i + 110
    np.random.seed(s)
    random.seed(s)
    NasHpoCell.cs.seed(s)
    t1 = time.time()
    tracker = parallel_regularized_evolution(profiler, args)
    output_path = os.path.join(args.output_path, args.benchmark, "re_nips",
                               'pop{}sam{}w{}m{}{}{}{}{}{}'.format(args.pop_size, args.sample_size, args.n_workers,
                                                           args.mutate_step,
                                                           "p"+ str(args.mutate_num) if args.predict_mutate else "",
                                                            "kn" + str(args.kill_num) if args.predict_kill else "",
                                                            "ka" + str(args.kill_alpha) if args.predict_kill else "",
                                                            "ks" + str(args.kill_ms) if args.predict_kill else "",
                                                            args.describe)
                                                  )

    os.makedirs(os.path.join(output_path), exist_ok=True)



    res, training_data = tracker.summary(profiler)
    res['best_model'] = ""

    with open(os.path.join(output_path, 'run_%d.json' % i), 'w') as f:
        print('Dumping result to {}'.format(os.path.join(output_path, 'run_%d.json' % i)))
        json.dump(res, f)


    # joblib.dump(training_data, os.path.join(output_path, 'train_%d.pkl' % i))


    # print('Dumping result to {}'.format(os.path.join(output_path, 'run_%d.pkl' % i)))
    # joblib.dump(tracker.serialize_models(), os.path.join(output_path, 'run_%d.pkl' % i))

    print('Trial {}'.format(i), (time.time() - t1))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--single', default=False, type=lambda x: (str(x).lower() == 'true'), nargs='?',
                        help='use multi-objective or not')
    parser.add_argument('--benchmark',  default='nasbench101', type=str, nargs='?', help='specifies the benchmark')
    parser.add_argument('--n_iters', default=5000, type=int, nargs='?',
                        help='number of iterations for optimization method')
    parser.add_argument('--output_path', default="./results", type=str, nargs='?',
                        help='specifies the path where the results will be saved')
    parser.add_argument('--method', default="re", choices=['re'])
    parser.add_argument('--data_dir', type=str, nargs='?',
                        help='specifies the path to the tabular data')
    parser.add_argument('--num_process', default=1, type=int, nargs='?', help='num processes')
    parser.add_argument('--num_trials', default=100, type=int, nargs='?', help='num trials')
    parser.add_argument('--pop_size', default=100, type=int, nargs='?', help='population size')
    parser.add_argument('--sample_size', default=10, type=int, nargs='?', help='sample size')
    parser.add_argument('--mutate_step', default=1.0, type=float, nargs='?', help='mutate step')
    parser.add_argument('--mutate_num', default=3, type=int, nargs='?', help='mutate num for predictor')
    parser.add_argument('--n_workers', default=16, type=int, nargs='?', help='number of workers')
    parser.add_argument('--predict_model', default=False, type=lambda x:(str(x).lower() == 'true'), nargs='?', help='use predictor or not')
    parser.add_argument('--describe', default='', type=str, nargs='?')
    parser.add_argument('--predict_mutate', default=False, type=lambda x: (str(x).lower() == 'true'), nargs='?',
                        help='use predictor or not')
    parser.add_argument('--predict_kill', default=False, type=lambda x: (str(x).lower() == 'true'), nargs='?',
                       help='use predictor or not')
    parser.add_argument('--kill_num', default=5, type=int, nargs='?', help='mutate num for predictor')
    parser.add_argument('--kill_alpha', default=0.1, type=float, nargs='?', help='mutate num for predictor')
    parser.add_argument('--kill_ms', default=1.0, type=float, nargs='?', help='mutate num for predictor')
    args = parser.parse_args()



    args.predict_model = args.predict_kill or args.predict_mutate

    benchmark_name = args.benchmark

    profiler = Profiler(benchmark_name)

    if args.single:
        run_trial(0, profiler, args)
    else:
        num_process = args.num_process
        for i in range(0, args.num_trials, num_process):
            process_list = []
            for j in range(i, i + num_process):
                p = Process(target=run_trial, args=(j, profiler, args))
                p.start()
                process_list.append(p)

            for p in process_list:
                p.join()
