import json
from time import time

import optuna
import pandas as pd

from catboost import CatBoostRegressor
from optuna.samplers import TPESampler
from sklearn.model_selection import cross_validate

optuna.logging.set_verbosity(optuna.logging.WARNING)

def get_params(trial):
    param = {
        "learning_rate": trial.suggest_loguniform("learning_rate", 1e-5, 1e0),
        "l2_leaf_reg": trial.suggest_loguniform("l2_leaf_reg", 1e-2, 1e0),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.01, 0.1),
        "depth": trial.suggest_int("depth", 1, 10),
        "boosting_type": trial.suggest_categorical("boosting_type", ["Ordered", "Plain"]),
        "bootstrap_type": trial.suggest_categorical("bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 2, 20),
        "one_hot_max_size": trial.suggest_int("one_hot_max_size", 3, 6),  
    }
    
    # Conditional Hyper-Parameters
    if param["bootstrap_type"] == "Bayesian":
        param["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0, 10)
    elif param["bootstrap_type"] == "Bernoulli":
        param["subsample"] = trial.suggest_float("subsample", 0.1, 1)
        
    return param
    
def get_objective(X: pd.DataFrame, y: pd.Series, clf_params={}, cv_params={}):
    def objective(trial):
        param = get_params(trial)
        model = CatBoostRegressor(**{**param, **clf_params})
        cv_results = cross_validate(model, X, y, **cv_params)
        return cv_results['test_score'].mean()
    
    return objective

def tune_catboost_cv(X: pd.DataFrame, y: pd.Series, seed=0, n_trials=30,
                     timeout=120, clf_params={}, cv_params={}, **kwargs):
    
    start = time()
    objective = get_objective(X, y, clf_params=clf_params, cv_params=cv_params)
    study = optuna.create_study(sampler=TPESampler(seed=seed), **kwargs)
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    finish = time()

    trial = study.best_trial
    print("\nBest Score: {}".format(trial.value))
    print("> Finished in: %.2fs" % (finish - start))
    print("> Number of completed trials: {}".format(len(study.trials)))
    
    print("> Best Params: ")
    print(json.dumps(trial.params, indent=4))
    print()
    
    best_params = dict(**study.best_trial.params, **clf_params)
    model = CatBoostRegressor(**best_params)
    cv_results = cross_validate(model, X, y, **cv_params)
    
    total_time = cv_results['fit_time'].sum() + cv_results['score_time'].sum()
    score = cv_results['test_score'].mean()
    print('> Done cross-validating in: %.2fs' % total_time)
    print('Refitted CV score:', score)
    
    return best_params, study
