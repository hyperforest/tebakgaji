from dataclasses import dataclass


@dataclass
class Config:
    # cv config
    seed: int = 0
    n_jobs: int = 1
    n_splits: int = 5
    scoring: str = 'neg_root_mean_squared_error'
    shuffle: bool = True

    # catboost config
    loss_function: str = 'RMSE'
    early_stopping_rounds: int = 200
    verbose: int = 0

    # optuna config
    direction: str = 'maximize'
    n_trials: int = 5
    timeout: int = 3600
    print_every: int = 2
