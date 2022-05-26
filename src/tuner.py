import json
from time import time
from typing import Callable, Dict

import optuna
import pandas as pd

from catboost import CatBoostRegressor
from optuna.samplers import TPESampler
from sklearn.model_selection import cross_validate

optuna.logging.set_verbosity(optuna.logging.WARNING)


def get_objective(X: pd.DataFrame, y: pd.Series,
                  params_func: Callable[[optuna.trial.Trial], Dict],
                  clf_params={}, cv_params={}
                  ) -> Callable[[optuna.trial.Trial], float]:
    def objective(trial):
        param = params_func(trial)
        model = CatBoostRegressor(**{**param, **clf_params})
        cv_results = cross_validate(model, X, y, **cv_params)
        return cv_results['test_score'].mean()

    return objective


def tune_catboost_cv(X: pd.DataFrame, y: pd.Series, 
                     params_func: Callable[[optuna.trial.Trial], Dict],
                     seed: int = 0, n_trials: int = 30, timeout: int = 120,
                     clf_params={}, cv_params={}, **kwargs):
    start = time()
    objective = get_objective(X, y, params_func, clf_params=clf_params,
                              cv_params=cv_params)
    study = optuna.create_study(sampler=TPESampler(seed=seed), **kwargs)
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    finish = time()

    best_trial = study.best_trial
    best_score = best_trial.value
    num_trials = len(study.trials)
    runtime = finish - start
    avg_runtime = runtime / num_trials
    print(f"> Number of completed trials: {num_trials}")
    print(f"> Finished in: {runtime:.1f}s ({avg_runtime:.1f}s/trial)")
    print("> Best Params: ")
    print(json.dumps(best_trial.params, indent=4))
    print(f"> Best Score: {best_score}")

    best_params = dict(**best_trial.params, **clf_params)

    return best_params, study
