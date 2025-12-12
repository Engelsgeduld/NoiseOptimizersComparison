import os

import mlflow
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from experiment.utils.preprocessing import BasePreprocessor


def _device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class ModelTester:
    def __init__(self, common_params: dict, save_predictions: bool = False):
        self.common_params = common_params
        self.device = torch.device(_device())
        self.save_predictions = save_predictions

    def test_model(
        self,
        test_signal: np.ndarray,
        start_sequence: np.ndarray,
        prep: BasePreprocessor,
        model: nn.Module | None = None,
        mlflow_run_id: str | None = None,
    ) -> dict:
        assert model is not None or mlflow_run_id is not None, "Нужно передать либо модель, либо mlflow_run_id"

        if model is None:
            model_uri = f"runs:/{mlflow_run_id}/model"
            model = mlflow.pytorch.load_model(model_uri)
        model.to(self.device)

        inference_data = prep.prepare_for_inference(test_signal, self.common_params["sequence_length"])

        X_test_model = inference_data["X_test_model"]
        y_test_unscaled = inference_data["y_test_unscaled"]
        reconstruction_aids = inference_data["reconstruction_aids"]

        onestep_loader = DataLoader(
            TensorDataset(
                torch.from_numpy(X_test_model).float().unsqueeze(-1),
                torch.from_numpy(np.zeros(len(X_test_model))).float().unsqueeze(-1),
            ),
            batch_size=self.common_params["batch_size"],
            shuffle=False,
        )

        predicted_processed, _ = self.predict_one_step(model, onestep_loader)
        preds_onestep_unscaled = prep.reconstruct_onestep(predicted_processed, reconstruction_aids)
        mse_onestep = mean_squared_error(y_test_unscaled, preds_onestep_unscaled)

        n_predict = len(test_signal)
        predicted_changes_auto = self.predict_autoregressive(model, start_sequence, n_predict)
        preds_auto_unscaled = prep.reconstruct_autoregressive(predicted_changes_auto)
        auto_mse = mean_squared_error(test_signal, preds_auto_unscaled)

        if mlflow_run_id:
            mlflow.log_metric("test_onestep_mse", mse_onestep)
            mlflow.log_metric("test_auto_mse", auto_mse)

            if self.save_predictions:
                onestep_preds_path = "preds_onestep.csv"
                df_onestep = pd.DataFrame({"true_values": y_test_unscaled, "predictions": preds_onestep_unscaled})
                df_onestep.to_csv(onestep_preds_path, index=False)
                mlflow.log_artifact(onestep_preds_path, "predictions")
                os.remove(onestep_preds_path)
                auto_preds_path = "preds_auto.csv"

                df_auto = pd.DataFrame({"true_values": test_signal, "predictions": preds_auto_unscaled})
                df_auto.to_csv(auto_preds_path, index=False)
                mlflow.log_artifact(auto_preds_path, "predictions")
                os.remove(auto_preds_path)

        return {
            "run_id": mlflow_run_id,
            "test_onestep_mse": mse_onestep,
            "test_auto_mse": auto_mse,
            "preds_onestep": preds_onestep_unscaled,
            "preds_auto": preds_auto_unscaled,
            "model": model,
        }

    def predict_one_step(self, model: nn.Module, dataloader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
        model.to(self.device)
        model.eval()
        predictions, ground_truth = [], []
        with torch.no_grad():
            for X_batch, y_batch in dataloader:
                outputs = model(X_batch.to(self.device))
                if outputs.ndim == 3:
                    outputs = outputs[:, -1, :]
                predictions.append(outputs.cpu().numpy())
                ground_truth.append(y_batch.cpu().numpy())
        return np.vstack(predictions).flatten(), np.vstack(ground_truth).flatten()

    def predict_autoregressive(self, model: nn.Module, start_sequence: np.ndarray, n_predict: int) -> np.ndarray:
        model.to(self.device)
        model.eval()
        predictions = []
        current_sequence = torch.from_numpy(start_sequence).float().unsqueeze(0).unsqueeze(-1).to(self.device)

        with torch.no_grad():
            for i in range(n_predict):
                next_pred_tensor = model(current_sequence)
                if next_pred_tensor.ndim == 3:
                    next_pred_tensor = next_pred_tensor[:, -1, :]

                pred_value = next_pred_tensor.cpu().numpy()[0, 0]
                predictions.append(pred_value)

                new_item_tensor = next_pred_tensor.unsqueeze(1)
                current_sequence = torch.cat((current_sequence[:, 1:, :], new_item_tensor), dim=1)

        predictions_array = np.array(predictions)
        return predictions_array
