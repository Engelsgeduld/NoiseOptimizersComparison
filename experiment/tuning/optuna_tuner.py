from typing import Any

import mlflow
import optuna

from configurations.configs import DatasetConfig, ExperimentConfig
from configurations.tuning_configs import TuningConfig
from experiment.model_trainer import StandardTrainer
from experiment.utils.preprocessing import ScalingAndDifferencingPreprocessor


class OptunaTuner:
    def __init__(
        self,
        tuning_config: TuningConfig,
        common_params: dict[str, Any],
        datasets: dict[str, DatasetConfig],
    ):
        self.tuning_config = tuning_config
        self.common_params = common_params
        self.datasets = datasets

    def _sample_params(self, trial: optuna.Trial, param_grid: dict[str, Any]) -> dict[str, Any]:
        sampled_params = {}
        for param_name, values in param_grid.items():
            sampled_params[param_name] = trial.suggest_categorical(param_name, values)
        return sampled_params

    def objective(self, trial: optuna.Trial) -> float:
        optimizer_params = self._sample_params(trial, self.tuning_config.optimizer_params_grid)
        criterion_params = self._sample_params(trial, self.tuning_config.criterion_params_grid)

        criterion = self.tuning_config.criterion_class(**criterion_params)

        total_val_loss = 0.0

        dataset_items = list(self.datasets.items())

        for i, (dataset_name, dataset_config) in enumerate(dataset_items):
            exp_conf = ExperimentConfig(
                exp_name=f"tuning_trial_{trial.number}",
                model=self.tuning_config.model,
                model_params=self.tuning_config.model_params,
                optimizer=self.tuning_config.optimizer,
                optimizer_params=optimizer_params,
                criterion=criterion,
                datasets=[dataset_name],
                preproc=ScalingAndDifferencingPreprocessor(),
                n_runs=1,
            )

            trainer = StandardTrainer(exp_conf=exp_conf, common_params=self.common_params, log_model=False)

            try:
                result = trainer.train_on_dataset(dataset_config.signal, dataset_name)
                val_loss = result["final_val_loss"]
                total_val_loss += val_loss

                current_avg_loss = total_val_loss / (i + 1)

                trial.report(current_avg_loss, step=i)

                if trial.should_prune():
                    raise optuna.TrialPruned()

            except optuna.TrialPruned:
                raise
            except Exception as e:
                print(f"Trial {trial.number} failed on dataset {dataset_name}: {e}")
                return float("inf")

        return total_val_loss / len(self.datasets)

    def run_tuning(self) -> optuna.Study:
        mlflow.set_experiment(self.tuning_config.study_name)

        with mlflow.start_run(run_name="optuna_tuning_session"):
            mlflow.log_params(
                {
                    "study_name": self.tuning_config.study_name,
                    "n_trials": self.tuning_config.n_trials,
                    "model": self.tuning_config.model,
                    "optimizer": self.tuning_config.optimizer,
                    "metric_direction": self.tuning_config.metric_direction,
                    **self.tuning_config.data_generation_config,
                }
            )

            for k, v in self.tuning_config.model_params.items():
                mlflow.log_param(f"model_fixed_{k}", v)

            study = optuna.create_study(
                direction=self.tuning_config.metric_direction,
                study_name=self.tuning_config.study_name,
                pruner=optuna.pruners.MedianPruner(),
            )

            study.optimize(self.objective, n_trials=self.tuning_config.n_trials)

            print("Best params:", study.best_params)
            print("Best value:", study.best_value)

            mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
            mlflow.log_metric("best_val_loss", study.best_value)

            return study
