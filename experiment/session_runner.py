import mlflow
import numpy as np
import pandas as pd

from configurations.configs import SessionConfig
from experiment.model_tester import ModelTester
from experiment.model_trainer import ModelTrainer


class SessionRunner:
    def __init__(self, session_config: SessionConfig, datasets: dict[str, np.ndarray], log_model: bool = True):
        self.session_config = session_config
        self.datasets = datasets
        self.log_model = log_model
        self.results: list = []

    def run(self) -> pd.DataFrame:
        mlflow.set_experiment(self.session_config.session_name)
        common_params = self.session_config.common_params

        for exp_conf in self.session_config.experiments:
            trainer = ModelTrainer(exp_conf, common_params, log_model=self.log_model)
            for dataset_name in exp_conf.datasets:
                result = trainer.train_on_dataset(self.datasets[dataset_name], dataset_name)

                tester = ModelTester(common_params)
                test_result = tester.test_model(
                    model=result.get("model"),
                    mlflow_run_id=result["mlflow_run_id"],
                    test_signal=self.session_config.test_signal,
                )
                result.update(test_result)

                self.results.append(result)

        return pd.DataFrame(self.results)
