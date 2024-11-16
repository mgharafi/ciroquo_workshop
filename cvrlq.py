from typing import Callable
import numpy as np
from cma.fitness_models import LQModel
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import minmax_scale
from scipy.stats import kendalltau
from cma import ff

def normalize(x):
    minx = np.min(x, axis=2).reshape(*(x.shape[:2]), 1)
    maxx = np.max(x, axis=2).reshape(*(x.shape[:2]), 1)
    denom = maxx - minx
    x = (x - minx) / denom
    return x

def custom_crossvalidation_set(size : int, test_size : int) -> list[int]:
    # sample random indices without replacement with normal distribution centered at 0
    return np.random.normal(0, 1, size).argsort()[:test_size]

def cross_validation_setup(U, Z, T, n_splits=10):
    
    train_data_size= len(Z)
    if n_splits > train_data_size:
        n_splits = train_data_size
    new_data_size = n_splits * int(np.floor(train_data_size / n_splits))
    T = T[:, :new_data_size]
    Z = Z[:new_data_size]
    # print(T.shape)
    train_idx = []
    test_idx = []

    for train_index, test_index in KFold(n_splits=n_splits).split(Z):
        train_idx += [train_index]
        test_idx += [test_index]
        # print(train_index, test_index)
    train_v, test_v = T[:, train_idx], T[:, test_idx]
    train_z, test_z = Z[train_idx], Z[test_idx]

    return train_v, test_v, train_z, test_z

def get_transformations_sorting(train_v, test_v, train_z, test_z):

    train_pinv = np.linalg.pinv(train_z)
    coefficients = np.einsum('bij,kbj->kbi', train_pinv, train_v)
    predictions = np.einsum('kbi,bmi->kbm', coefficients, test_z)
    correlation = [
                    [
                        kendalltau(pp, tt).statistic 
                        for pp, tt in zip(pred, test_vv)
                    ] 
                    for pred, test_vv in zip(predictions, test_v)
                ]
    correlation = np.mean(np.asarray(correlation), axis=1)
    return np.argsort(correlation)[::-1]


def cma_filter(attr : str) -> bool:
    return (
            not attr.startswith(('__', 'grad'))
            and attr not in [
                'somenan', 'rot', 'flat', 'epslow', 'leadingones', 'normalSkew', 'BBOB', 'fun_as_arg', 
                # 'cornerelli',
                # 'cornerellirot',
                # 'cornersphere',
                # 'elliwithoneconstraint',
                # 'lincon',
                # 'lineard',
                'sphere_pos',
                # 'spherewithnconstraints',
                # 'spherewithoneconstraint',
                'binary_foffset',
                'binary_optimum_interval',
                'evaluations',
            ]
            # and isinstance(getattr(ff, attr), MethodType)
            )


def cma_problem_list() -> list[str]:
    return [attr for attr in dir(ff) if cma_filter(attr)]

def get_problem(attr : str) -> Callable:
    values : dict[tuple[float], float] = {}
    fun : Callable= getattr(ff, attr)
    def inner(x : list[float]) -> float:
        if tuple(x) not in values:
            try:
                val : float = fun(x)
            except Exception as e:
                print(attr, e)
            values[tuple(x)] = val
        return values[tuple(x)]
    return inner


CMA_PROBLEMS = list(map(get_problem, cma_problem_list()))

def cma_ff_transf(X : list[list[float]], CMA_PROBLEMS : list[Callable]) -> list[list[float]]:
    ys : list[list[float]] = []
    for fun in CMA_PROBLEMS:
        try:
            y : list[float] = [fun(x) for x in X]
        except Exception as e:  # noqa: F841
            continue
        if np.isnan(y).any():
            continue
        if np.isinf(y).any():
            continue
        y = minmax_scale(y)
        y = np.sort(y)
        ys.append(y.flatten())
    return np.asarray(ys, dtype=float)

def repeat(arr, reps): return np.tile(arr, reps).reshape(reps, -1)

def transformations(S, es, m):
    distance_to_min = np.linalg.norm(S - S[0], axis=1)
    mean = es.mean
    mahalanobis = np.asarray([es.mahalanobis_norm(x - mean) for x in S])
    D = repeat(distance_to_min, m // 2)
    M = repeat(mahalanobis, m // 2)
    T = np.sort(D, axis=1) ** np.abs(np.random.normal(0, 1, (m // 2, 1)))
    T = np.vstack((T, np.sort(M, axis=1) ** np.abs(np.random.normal(0, 1, (m // 2, 1)))))

    minT = T[:, 0].reshape(-1, 1)
    maxT = T[:, -1].reshape(-1, 1)

    T -= minT
    T /= maxT - minT

    return np.sort(distance_to_min), T

class RankLQModel(LQModel):

    def __init__(self, use_cma_transformation=False,**super_args):
        super().__init__(**super_args)
        self.use_cma_transformation = use_cma_transformation
    @property
    def coefficients(self):
        """model coefficients that are linear in self.expand(.)"""
        if self._coefficients_count < self.count:
            self._coefficients_count = self.count
            self._coefficients = self.compute_coefficients(self.pinv, self.Y) #np.dot(self.pinv, self.weighted_array(self.Y))
            self.logger.push()  # use logging_trace attribute and xopt
            self.log_eigenvalues.push()
        
        return self._coefficients
    
    def compute_coefficients(self, pinv, Y):
        n = self.settings.n_for_model_building(self)
        return self.ensemble_model(pinv, Y, n)
    
    def ensemble_model(self, pinv, Y, n):

        size = self.settings.n_for_model_building(self)
        idx = self.settings.sorted_index(self)  # by default argsort(self.Y)
        
        if size < self.size:
            idx = idx[:size]

        U = self.X[idx]
        # shift to current optimum
        U -= U[0]
        V = self.Y[idx]
        Z = self.Z[idx]
        
        # es = self.es
        # H, T = transformations(U, es, 100)
        if self.use_cma_transformation:
            C = cma_ff_transf(U, CMA_PROBLEMS)
            T = np.vstack((V, np.linspace(0,1,size), C))
        else:
            T = np.vstack((V, np.linspace(0, 1, size), V**.1, V**.5, V**.9, V**2, V**4))

        sorted_models = get_transformations_sorting(*cross_validation_setup(U,Z,T))
        
        if not self.weighted:
            return np.linalg.pinv(Z) @ T[sorted_models[0]]
        else:
            E = T[sorted_models[0]]
            A = (self.sorted_weights(n) * np.asarray(E).T).T
            return pinv @ A