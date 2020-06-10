import numpy as np
import matplotlib.pyplot as plt
class Tracker:

    def __init__(self):

        self.evaluated_models = []

        self.y_star_valid = 0.04944576819737756  # lowest mean validation error
        self.y_star_test = 0.056824247042338016  # lowest mean test error

    def record_model(self, m):

        if isinstance(m, list):
            for m1 in m:
                if not m1.es:
                    self.evaluated_models.append(m1)
        else:
            if not m.es:
                self.evaluated_models.append(m)

    def summary(self, profiler):
        sorted(self.evaluated_models, key=lambda model:model.finished_time)
        regret_validation = []
        regret_test = []
        params = []
        costs = []
        runtime = []
        distances = []
        predictor_X_data = []
        predictor_Y_data = []
        best_model = None
        inc_valid = np.inf
        inc_test = np.inf

        inc_params = np.inf
        inc_distance = np.inf
        inc_cost = np.inf
        all_points = []
        for idx, model in enumerate(self.evaluated_models):

            if model.training_time == 0:  # invalid model
                continue

            val_error = 1 - model.accuracy
            mean_test_error = 1 - model.test_accuracy

            if inc_valid > val_error:
                inc_valid = val_error

                inc_test = mean_test_error
                inc_cost = model.cost
                inc_params = model.params
                inc_distance = model.distance_from_root()
                best_model = model
            regret_validation.append(float(inc_valid - self.y_star_valid))
            regret_test.append(float(inc_test - self.y_star_test))
            params.append(inc_params)
            costs.append(inc_cost)
            distances.append(inc_distance)

            # x axis
            runtime.append(float(model.finished_time))
            all_points.append([model.cost, val_error, mean_test_error])

            predictor_X_data.append(model.serialize())
            predictor_Y_data.append(model.accuracy)


        pareto_points = all_points

        res = dict()
        res['regret_validation'] = regret_validation
        res['regret_test'] = regret_test
        res['runtime'] = runtime
        res['params'] = params
        res['costs'] = costs
        res['distance'] = distances
        res['pareto'] = pareto_points
        if best_model:
            res['best_model'] = best_model.arch.serialize()
        return res, (predictor_X_data, predictor_Y_data)




    def serialize_models(self):
        seralized_models = []

        for model in self.evaluated_models:
            seralized_models.append(model.serialize())

        return seralized_models
