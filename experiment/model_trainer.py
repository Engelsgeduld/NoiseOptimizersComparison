from dataclasses import asdict

import mlflow
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from configurations.configs import ExperimentConfig
from configurations.models_optimizers import MODELS_MAP, OPTIMIZERS_MAP
from experiment.utils.sequences_creator import create_sequences


class ModelTrainer:
    def __init__(self, exp_conf: ExperimentConfig, common_params: dict, log_model: bool = True):
        self.exp_conf = exp_conf
        self.common_params = common_params
        self.log_model = log_model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.models_map = MODELS_MAP
        self.optimizers_map = OPTIMIZERS_MAP

    def train_on_dataset(self, signal: np.ndarray, dataset_name: str) -> dict:
        run_name = f"{self.exp_conf.exp_name}_on_{dataset_name}"

        with mlflow.start_run(run_name=run_name) as run:
            mlflow.log_params({**self.common_params, **asdict(self.exp_conf), "dataset": dataset_name})

            X, y = create_sequences(signal, self.common_params["sequence_length"])
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
            train_loader, val_loader = self._make_dataloaders(X_train, y_train, X_val, y_val)

            model = self._make_model()
            optimizer = self._make_optimizer(model)
            criterion = nn.MSELoss()

            train_losses, val_losses = [], []
            for epoch in range(self.common_params["epochs"]):
                train_loss = self._train_one_epoch(model, train_loader, optimizer, criterion)
                val_loss = self._validate_one_epoch(model, val_loader, criterion)
                mlflow.log_metrics({"train_loss": train_loss, "val_loss": val_loss}, step=epoch)
                train_losses.append(train_loss)
                val_losses.append(val_loss)

            if self.log_model:
                model_artifact_name = f"model-{self.exp_conf.exp_name}-{dataset_name}"
                mlflow.pytorch.log_model(
                    pytorch_model=model, artifact_path="model", registered_model_name=model_artifact_name
                )

            return {
                "exp_name": self.exp_conf.exp_name,
                "dataset": dataset_name,
                "mlflow_run_id": run.info.run_id,
                "model": model if not self.log_model else None,
                "final_train_loss": train_losses[-1],
                "final_val_loss": val_losses[-1],
            }

    def _make_model(self) -> nn.Module:
        ModelClass = self.models_map[self.exp_conf.model]
        return ModelClass(**self.exp_conf.model_params).to(self.device)

    def _make_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        OptClass = self.optimizers_map[self.exp_conf.optimizer]
        return OptClass(model.parameters(), **self.exp_conf.optimizer_params)

    def _make_dataloaders(
        self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray
    ) -> tuple[DataLoader, DataLoader]:
        def make_ds(X: np.ndarray, y: np.ndarray) -> TensorDataset:
            return TensorDataset(torch.from_numpy(X).float().unsqueeze(-1), torch.from_numpy(y).float().unsqueeze(-1))

        train_loader = DataLoader(make_ds(X_train, y_train), batch_size=self.common_params["batch_size"], shuffle=True)
        val_loader = DataLoader(make_ds(X_val, y_val), batch_size=self.common_params["batch_size"], shuffle=False)
        return train_loader, val_loader

    def _train_one_epoch(
        self, model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, criterion: nn.Module
    ) -> float:
        model.train()
        total = 0
        for X, y in loader:
            X, y = X.to(self.device), y.to(self.device)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()
            total += loss.item()
        return total / len(loader)

    def _validate_one_epoch(self, model: nn.Module, loader: DataLoader, criterion: nn.Module) -> float:
        model.eval()
        total = 0
        with torch.no_grad():
            for X, y in loader:
                X, y = X.to(self.device), y.to(self.device)
                loss = criterion(model(X), y)
                total += loss.item()
        return total / len(loader)
