import mlflow
import numpy as np
import torch
from sklearn.metrics import mean_squared_error
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from experiment.utils.sequences_creator import create_sequences


class ModelTester:
    def __init__(self, common_params: dict):
        self.common_params = common_params
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_model(
        self, test_signal: np.ndarray, model: nn.Module | None = None, mlflow_run_id: str | None = None
    ) -> dict:
        assert model is not None or mlflow_run_id is not None, "Нужно передать либо модель, либо mlflow_run_id"

        if model is None:
            model_uri = f"runs:/{mlflow_run_id}/model"
            model = mlflow.pytorch.load_model(model_uri)
        model.to(self.device)

        X_test, y_test = create_sequences(test_signal, self.common_params["sequence_length"])
        test_loader = DataLoader(
            TensorDataset(
                torch.from_numpy(X_test).float().unsqueeze(-1), torch.from_numpy(y_test).float().unsqueeze(-1)
            ),
            batch_size=self.common_params["batch_size"],
            shuffle=False,
        )

        preds, true_preds = self.predict_one_step(model, test_loader)
        mse_onestep = mean_squared_error(true_preds, preds)

        start_sequence = test_signal[: self.common_params["sequence_length"]]
        n_predict = len(test_signal) - self.common_params["sequence_length"]
        auto_preds = self.predict_autoregressive(model, start_sequence, n_predict)
        auto_mse = mean_squared_error(test_signal[self.common_params["sequence_length"] :], auto_preds)

        if mlflow_run_id:
            with mlflow.start_run(run_id=mlflow_run_id):
                mlflow.log_metric("test_onestep_mse", mse_onestep)
                mlflow.log_metric("test_auto_mse", auto_mse)

        return {
            "run_id": mlflow_run_id,
            "test_onestep_mse": mse_onestep,
            "test_auto_mse": auto_mse,
            "preds_onestep": preds,
            "preds_auto": auto_preds,
            "model": model,
        }

    def predict_one_step(self, model: nn.Module, dataloader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
        model.to(self.device)
        model.eval()
        predictions, ground_truth = [], []
        with torch.no_grad():
            for X_batch, y_batch in dataloader:
                outputs = model(X_batch.to(self.device))
                predictions.append(outputs.cpu().numpy())
                ground_truth.append(y_batch.cpu().numpy())
        return np.vstack(predictions).flatten(), np.vstack(ground_truth).flatten()

    def predict_autoregressive(self, model: nn.Module, start_sequence: np.ndarray, n_predict: int) -> np.ndarray:
        model.to(self.device)
        model.eval()
        predictions = []
        current_sequence = torch.from_numpy(start_sequence).float().unsqueeze(0).unsqueeze(-1).to(self.device)

        with torch.no_grad():
            for _ in range(n_predict):
                next_pred_tensor = model(current_sequence)
                predictions.append(next_pred_tensor.cpu().numpy()[0, 0])
                new_item_tensor = next_pred_tensor.unsqueeze(1)
                current_sequence = torch.cat((current_sequence[:, 1:, :], new_item_tensor), dim=1)

        return np.array(predictions)
