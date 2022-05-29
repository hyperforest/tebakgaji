import json
import pickle

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_validate

from modules.data import clean_city, clean_data, clean_role, read_raw
from modules.config import Config

DATA_PATH = './datasets/data.html'
CONFIG_PATH = './config/train_config.json'
BEST_PARAMS_PATH = './config/best_hyperparam.json'

MODEL_PATH = './model/catboost_model.pkl'
VECTORIZER_PATH = './model/count_vectorizer.pkl'

def main():
    # prepare config
    with open(CONFIG_PATH, 'r') as file:
        config = Config(**json.load(file))

    print('> Train config:')
    print(json.dumps(config.__dict__, indent=4))

    with open(BEST_PARAMS_PATH, 'r') as file:
        best_params = json.load(file)

    print('> Best hyperparameters:')
    print(json.dumps(best_params, indent=4))

    # prepare data
    raw_data = read_raw(DATA_PATH)
    print('> Data shape (raw):', raw_data.shape)

    df = clean_data(raw_data)
    df.city = df.apply(clean_city, axis=1)
    df.role = df.apply(clean_role, axis=1)
    print('> Data shape (clean):', df.shape)

    vec = CountVectorizer(binary=True)
    vec.fit(df.role)
    counts = pd.DataFrame(
        vec.transform(df.role).toarray(),
        columns=vec.get_feature_names_out()
    )

    cols = ['company', 'city', 'years_of_exp']
    X, y = pd.concat([counts, df[cols]], axis=1), df.salary
    cat_features = np.where(X.columns != 'years_of_exp')[0]
    print('> Data shape (train set):', X.shape)

    # prepare training params
    clf_params = dict(random_seed=config.seed,
                      loss_function=config.loss_function,
                      cat_features=cat_features)

    cv = KFold(n_splits=config.n_splits, shuffle=config.shuffle,
              random_state=config.seed)

    fit_params = dict(early_stopping_rounds=config.early_stopping_rounds,
                      verbose=config.verbose)

    cv_params = dict(cv=cv, fit_params=fit_params, scoring=config.scoring,
                     return_train_score=True)

    model = CatBoostRegressor(**{**clf_params, **best_params})
    cv_results = cross_validate(model, X, y, **cv_params)
    
    train_score = cv_results['train_score'].mean()
    test_score = cv_results['test_score'].mean()
    cv_time = cv_results['fit_time'].sum() + cv_results['score_time'].sum()

    model = CatBoostRegressor(**{**clf_params, **best_params})
    model.fit(X, y, **fit_params)
    y_pred = model.predict(X)
    refit_score = np.sqrt(mean_squared_error(y, y_pred))

    print(f'> Cross-validation time: {cv_time:.1f}s')
    print(f'> CV score (train set): {train_score:.4f}')
    print(f'> CV score (validation set): {test_score:.4f}')
    print(f'> Refit score: {refit_score:.4f}')

    with open(MODEL_PATH, 'wb') as file:
        pickle.dump(model, file)

    print(f'> Model saved to {MODEL_PATH}')

    with open(VECTORIZER_PATH, 'wb') as file:
        pickle.dump(vec, file)

    print(f'> Vectorizer saved to {VECTORIZER_PATH}')

if __name__ == '__main__':
    main()
