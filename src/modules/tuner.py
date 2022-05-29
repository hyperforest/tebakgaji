import json
from datetime import datetime
from typing import Callable

import optuna


def get_logging_every_n_and_best_trial(print_every: int = 5):
    def log_every_n_and_best_trial(study, frozen_trial):
        previous_best_value = study.user_attrs.get("previous_best_value", None)
        if previous_best_value != study.best_value:
            study.set_user_attr("previous_best_value", study.best_value)
            msg = " --- ".join([
                f"[LOGGING] New best found (trial {frozen_trial.number})",
                f"Runtime: {frozen_trial.duration.total_seconds():.1f}s",
                f"Score: {frozen_trial.value:.4f}"
            ])
            print(msg)
        
        total_trials = len(study.trials)
        if total_trials % print_every == 0:
            duration = sum(map(
                lambda trial: trial.duration.total_seconds(), study.trials
                ))
            avg_duration = duration / total_trials
            score = study.trials[-1].value
            best_trial_num = study.best_trial.number
            best_value = study.best_value

            msg = ' --- '.join([
                f"[LOGGING] {total_trials} trials done",
                f"Avg. runtime: {avg_duration:.1f}s/trial",
                f"Latest score: {score:.4f}",
                f"Best score (in trial {best_trial_num}): {best_value:.4f}"
            ])
            print(msg)
    
    return log_every_n_and_best_trial


def tune(
    objective: Callable[[optuna.trial._trial.Trial], float],
    study_params={},
    opt_params={},
    **kwargs
):
    start = datetime.now()
    study = optuna.create_study(**study_params)
    study.optimize(objective, **opt_params)
    finish = datetime.now()

    best_trial = study.best_trial
    best_score = best_trial.value
    num_trials = len(study.trials)
    runtime = finish - start
    str_runtime = str(runtime).split('.')[0]
    avg_runtime = (finish - start).total_seconds() / num_trials

    print()
    print(f"> Number of completed trials: {num_trials}")
    print(f"> Finished in: {str_runtime} ({avg_runtime:.1f}s/trial)")
    print("> Best params:")
    print(json.dumps(best_trial.params, indent=4))
    print(f"> Best score (in trial {best_trial.number}): {best_score}")

    return best_trial.params, study
