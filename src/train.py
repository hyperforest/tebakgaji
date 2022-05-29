import json
import pickle
from warnings import simplefilter

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import KFold, cross_validate

from modules.config import Config
from modules.data import clean_city, clean_data, clean_role, read_raw
from modules.tuner import get_logging_every_n_and_best_trial, tune

optuna.logging.set_verbosity(optuna.logging.WARNING)
simplefilter("ignore", category=RuntimeWarning)

DATA_PATH = './datasets/data.html'
CONFIG_PATH = './config/train_config.json'

MODEL_PATH = './model/catboost_model.pkl'
VECTORIZER_PATH = './model/count_vectorizer.pkl'

def get_params_func(**kwargs):
    one_hot_max_size_choice = kwargs.get(
        'one_hot_max_size_choice', [2, 3, 4, 5, 6]
    )
    
    def params_func(trial):
        param = {
            "depth": trial.suggest_int("depth", 4, 10),
            "boosting_type": trial.suggest_categorical(
                "boosting_type", ["Ordered", "Plain"]
            ),
            "bootstrap_type": trial.suggest_categorical(
                "bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]
            ),
            "colsample_bylevel": trial.suggest_float(
                "colsample_bylevel", 0.01, 1.0
            ),
            "n_estimators": trial.suggest_categorical(
                "n_estimators", [500, 1000, 1500, 2000]
            ),
            "one_hot_max_size": trial.suggest_categorical(
                "one_hot_max_size", one_hot_max_size_choice
            )
        }

        # Conditional Hyper-Parameters
        if param["bootstrap_type"] == "Bayesian":
            param["bagging_temperature"] = trial.suggest_float(
                "bagging_temperature", 1.0, 20.0
            )
        elif param["bootstrap_type"] == "Bernoulli":
            param["subsample"] = trial.suggest_float("subsample", 0.33, 1.0)
        else:
            param["subsample"] = trial.suggest_float("subsample", 0.6, 1.0)
        
        return param
    return params_func

def main():
    # prepare config
    with open(CONFIG_PATH, 'r') as file:
        config = Config(**json.load(file))

    print('> Train config:')
    print(json.dumps(config.__dict__, indent=4))

    # prepare data
    raw_data = read_raw(DATA_PATH)
    print('> Data shape (raw):', raw_data.shape)

    df = clean_data(raw_data)
    df.city = df.apply(clean_city, axis=1)
    df.role = df.apply(clean_role, axis=1)

    # extract token count
    vec = CountVectorizer(binary=True)
    vec.fit(df.role)
    counts = pd.DataFrame(
        vec.transform(df.role).toarray(),
        columns=vec.get_feature_names_out()
    )

    # prepare data
    cols = ['company', 'city', 'years_of_exp']
    X, y = pd.concat([counts, df[cols]], axis=1), df.salary
    cat_features = np.where(X.columns != 'years_of_exp')[0]
    print('Data shape:', X.shape)

    # prepare training params
    log_every_n_and_best_trial = get_logging_every_n_and_best_trial(
        print_every=config.print_every
    )
    callbacks = [log_every_n_and_best_trial]

    clf_params = dict(random_seed=config.seed,
                      loss_function=config.loss_function,
                      cat_features=cat_features)

    cv = KFold(n_splits=config.n_splits, shuffle=config.shuffle,
              random_state=config.seed)

    fit_params = dict(early_stopping_rounds=config.early_stopping_rounds,
                      verbose=config.verbose)

    cv_params = dict(cv=cv, fit_params=fit_params, scoring=config.scoring)

    opt_params = dict(n_trials=config.n_trials, timeout=config.timeout,
                      callbacks=callbacks)

    pruner = optuna.pruners.HyperbandPruner()
    sampler = optuna.samplers.TPESampler(seed=config.seed)
    study_params = dict(sampler=sampler, pruner=pruner,
                        direction=config.direction)

    # start training
    one_hot_max_size_choice = X.drop('years_of_exp', axis=1).nunique()
    one_hot_max_size_choice = list(set(one_hot_max_size_choice))
    params_func = get_params_func(
        one_hot_max_size_choice=one_hot_max_size_choice
    )

    def objective(trial):
        param = params_func(trial)
        model = CatBoostRegressor(**{**param, **clf_params})
        cv_results = cross_validate(model, X, y, **cv_params)
        return cv_results['test_score'].mean()

    best_params, study = tune(objective, study_params=study_params,
                            opt_params=opt_params)


    model = CatBoostRegressor(**{**best_params, **clf_params})
    model.fit(X, y, **fit_params)
    with open(MODEL_PATH, 'wb') as file:
        pickle.dump(model, file)

    print(f'> Model saved to {MODEL_PATH}')

    with open(VECTORIZER_PATH, 'wb') as file:
        pickle.dump(vec, file)

    print(f'> Vectorizer saved to {VECTORIZER_PATH}')

if __name__ == '__main__':
    main()
