from typing import Any

import mlflow
import numpy as np
import pandas as pd
from scipy import stats

from configurations.configs import DatasetConfig, ExperimentConfig, SessionConfig
from configurations.politics import TRAINERS_MAP
from experiment.model_tester import ModelTester


class SessionRunner:
    def __init__(
        self,
        session_config: SessionConfig,
        datasets: dict[str, DatasetConfig],
        log_model: bool = True,
        save_predictions: bool = False,
    ):
        self.session_config = session_config
        self.datasets = datasets
        self.log_model = log_model
        self.save_predictions = save_predictions
        self.results: list = []

    def run(self) -> pd.DataFrame:
        mlflow.set_experiment(self.session_config.session_name)

        for exp_conf in self.session_config.experiments:
            for dataset_name in exp_conf.datasets:
                aggregated_result = self._run_experiment(exp_conf, dataset_name)
                self.results.append(aggregated_result)

        return pd.DataFrame(self.results)

    def _run_experiment(self, exp_conf: ExperimentConfig, dataset_name: str) -> dict[str, Any]:
        parent_run_name = f"{exp_conf.exp_name}_on_{dataset_name}"
        with mlflow.start_run(run_name=parent_run_name) as parent_run:
            dataset_config = self.datasets[dataset_name]

            params_to_log = {}
            params_to_log.update(self.session_config.common_params)

            params_to_log["exp_name"] = exp_conf.exp_name
            params_to_log["politic"] = exp_conf.politic
            params_to_log["politic_params"] = exp_conf.politic_params
            params_to_log["dataset"] = dataset_name
            params_to_log["model"] = exp_conf.model
            params_to_log["optimizer"] = exp_conf.optimizer
            params_to_log["model_params"] = exp_conf.model_params
            params_to_log["optimizer_params"] = exp_conf.optimizer_params
            params_to_log.update(dataset_config.metadata)

            mlflow.log_params(params_to_log)
            mlflow.set_tag("parent_run", "True")

            trial_results = []
            for i in range(exp_conf.n_runs):
                result = self._run_single_trial(exp_conf, dataset_config, run_number=i + 1, base_params=params_to_log)
                trial_results.append(result)

            aggregated_metrics = self._aggregate_and_log_results(trial_results)

            final_result = {
                "exp_name": exp_conf.exp_name,
                "dataset": dataset_name,
                "n_runs": exp_conf.n_runs,
                "parent_run_id": parent_run.info.run_id,
                **aggregated_metrics,
            }
            return final_result

    def _run_single_trial(
        self,
        exp_conf: ExperimentConfig,
        dataset_config: DatasetConfig,
        run_number: int,
        base_params: dict,
    ) -> dict[str, Any]:
        with mlflow.start_run(run_name=f"run_{run_number}", nested=True) as child_run:
            mlflow.log_params({**base_params, "dataset": dataset_config.name})
            mlflow.set_tag("run_number", str(run_number))

            TrainerClass = TRAINERS_MAP[exp_conf.politic]
            trainer = TrainerClass(
                exp_conf=exp_conf,
                common_params=self.session_config.common_params,
                log_model=self.log_model,
                **exp_conf.politic_params,
            )

            train_result = trainer.train_on_dataset(dataset_config.signal, dataset_config.name)
            start_seq = train_result.get("start_sequence")

            if start_seq is None:
                raise ValueError("Trainer did not return start_sequence")

            tester = ModelTester(self.session_config.common_params, save_predictions=self.save_predictions)
            test_result = tester.test_model(
                model=train_result.get("model"),
                mlflow_run_id=child_run.info.run_id,
                test_signal=self.session_config.test_signal,
                prep=exp_conf.preproc,
                start_sequence=np.array(start_seq),
            )

            del test_result["model"]
            del train_result["model"]
            if "start_sequence" in train_result:
                del train_result["start_sequence"]

            return {**train_result, **test_result}

    def _aggregate_and_log_results(self, trial_results: list[dict[str, Any]]) -> dict[str, Any]:
        if not trial_results:
            return {}

        df_results = pd.DataFrame(trial_results)
        metrics_to_agg = ["final_train_loss", "final_val_loss", "test_onestep_mse", "test_auto_mse"]
        aggregated_metrics = {}

        for metric in metrics_to_agg:
            if metric not in df_results.columns:
                continue

            values = df_results[metric].dropna()
            if len(values) == 0:
                continue

            mean = values.mean()
            std = values.std()

            aggregated_metrics[f"avg_{metric}"] = mean
            aggregated_metrics[f"std_{metric}"] = std

            if len(values) > 1:
                ci = stats.t.interval(0.95, len(values) - 1, loc=mean, scale=stats.sem(values))
                aggregated_metrics[f"ci95_low_{metric}"] = ci[0]
                aggregated_metrics[f"ci95_high_{metric}"] = ci[1]

        if aggregated_metrics:
            mlflow.log_metrics(aggregated_metrics)

        return aggregated_metrics
