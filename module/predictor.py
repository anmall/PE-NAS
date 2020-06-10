from module.cell import Cell
import numpy as np
import lightgbm as lgb

from sklearn.neighbors import KNeighborsRegressor

class Predictor(object):

    def __init__(self, model_name="lgb"):
        self.model_name = model_name
        self.model = self._init_single_model(model_name)


    def _init_single_model(self, model):

        if model == 'lgb':
            return lgb.LGBMRegressor()

        return None


    def predict(self, X):
        return self.model.predict(X)


    def fit(self, dp):
        if self.model_name == 'lgb':
            estimator = self.model
            param_grid = {
                'n_estimators': [50, 100, 150, 200, 250, 300, 400, 500],
                'num_leaves': [5, 10, 30, 50, 100, 150, 200],
            }
            gbm_gs = GridSearchCV(estimator, param_grid, cv=5, n_jobs=10, verbose=1)
            gbm_gs.fit(dp.X, dp.y)
            print("Best param is", gbm_gs.best_params_)
            self.model = gbm_gs.best_estimator_

        self.model.fit(dp.X, dp.y)




class DataPreparer(object):

    def __init__(self, benchmark_name):
        self.X = None
        self.y = None
        self.benchmark_name = benchmark_name


    def load_evaluated_models(self, evaluated_models):

        X = []
        y = []

        for i, m in enumerate(evaluated_models):
            m = m.serialize()
            X.append(self.json_model_feature(m))
            y.append(m['acc'])

        if self.X is not None and self.y is not None:
            self.X = np.concatenate([self.X, np.array(X)], axis=0)
            self.y = np.concatenate([self.y, np.array(y)], axis=0)
        else:
            self.X = np.array(X)
            self.y = np.array(y)


    def json_model_feature(self, m):
        arch_encoding = self._arch_encoding(m)
        other_feature = self._arch_runtime_feature(m)
        return arch_encoding + other_feature



    def _arch_runtime_feature(self, m):
        feature1 = [ m['params'] , m['cost'] ]
        return feature1


    def _arch_encoding(self, m):
        if self.benchmark_name.startswith('nasbench101'):
            arch_encoding = Cell(np.array(m['arch']['matrix']), m['arch']['ops']).encode_cell()
            return arch_encoding

    def arch_encoding_from_arch(self, arch):
        return Cell(np.array(arch['matrix']), arch['ops']).encode_cell()




