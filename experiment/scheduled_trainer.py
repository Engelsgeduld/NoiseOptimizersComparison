import math

import mlflow
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader

from experiment.model_trainer import StandardTrainer
from experiment.utils.sequences_creator import create_sequences


class ScheduledSamplingTrainer(StandardTrainer):
    def _get_teacher_forcing_ratio(self, epoch: int) -> float:
        decay_type = self.common_params.get("decay_type", "inverse_sigmoid")
        k = self.common_params.get("k", 10)
        min_ratio = self.common_params.get("min_ratio", 0.0)

        if decay_type == "linear":
            ratio = max(min_ratio, 1.0 - k * epoch)
        elif decay_type == "exponential":
            ratio = max(min_ratio, math.exp(-k * epoch))
        elif decay_type == "inverse_sigmoid":
            ratio = max(min_ratio, k / (k + math.exp(epoch / k)))
        else:
            ratio = 1.0

        return ratio

    def _train_one_epoch(self, model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer) -> float:
        model.train()
        total = 0

        ratio = self._get_teacher_forcing_ratio(self.current_epoch)
        mlflow.log_metric("teacher_forcing_ratio", ratio, step=self.current_epoch)

        for X, y in loader:
            X, y = X.to(self.device), y.to(self.device)

            targets = torch.cat((X[:, 1:, :], y.unsqueeze(1)), dim=1)

            optimizer.zero_grad()

            outputs = model(X, teacher_forcing_ratio=ratio)

            loss = self.exp_conf.criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total += loss.item()

        return total / len(loader)

    def train_on_dataset(self, signal: np.ndarray, dataset_name: str) -> dict:
        train_signal, val_signal = train_test_split(signal, test_size=0.2, shuffle=False)

        self.exp_conf.preproc.fit(train_signal)

        train_signal_processed = self.exp_conf.preproc.transform(train_signal)
        val_signal_processed = self.exp_conf.preproc.transform(val_signal)

        X_train, y_train = create_sequences(train_signal_processed, self.common_params["sequence_length"])
        X_val, y_val = create_sequences(val_signal_processed, self.common_params["sequence_length"])

        train_loader, val_loader = self._make_dataloaders(X_train, y_train, X_val, y_val)

        model = self._make_model()
        optimizer = self._make_optimizer(model)

        train_losses, val_losses = [], []
        for epoch in range(self.common_params["epochs"]):
            self.current_epoch = epoch
            train_loss = self._train_one_epoch(model, train_loader, optimizer)
            val_loss = self._validate_one_epoch(model, val_loader)
            mlflow.log_metrics({"train_loss": train_loss, "val_loss": val_loss}, step=epoch)
            train_losses.append(train_loss)
            val_losses.append(val_loss)

        if self.log_model:
            model_artifact_name = f"model-{self.exp_conf.exp_name}-{dataset_name}"
            mlflow.pytorch.log_model(
                pytorch_model=model, artifact_path="model", registered_model_name=model_artifact_name
            )

        full_signal_processed = self.exp_conf.preproc.transform(signal)
        start_sequence = full_signal_processed[-self.common_params["sequence_length"] :]

        return {
            "model": model,
            "final_train_loss": train_losses[-1],
            "final_val_loss": val_losses[-1],
            "start_sequence": start_sequence,
        }
