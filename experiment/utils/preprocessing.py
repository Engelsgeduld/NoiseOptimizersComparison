from abc import ABC, abstractmethod

import numpy as np
from sklearn.preprocessing import MinMaxScaler

from experiment.utils.sequences_creator import create_sequences


class BasePreprocessor(ABC):
    @abstractmethod
    def fit(self, train_signal: np.ndarray) -> "BasePreprocessor":
        pass

    @abstractmethod
    def transform(self, signal: np.ndarray) -> np.ndarray:
        pass

    def fit_transform(self, train_signal: np.ndarray) -> np.ndarray:
        return self.fit(train_signal).transform(train_signal)

    @abstractmethod
    def reconstruct_autoregressive(self, predicted_processed: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def reconstruct_onestep(self, predicted_processed: np.ndarray, aids: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def prepare_for_inference(self, test_signal: np.ndarray, sequence_length: int) -> dict:
        pass


class IdentityPreprocessor(BasePreprocessor):
    def fit(self, train_signal: np.ndarray) -> "IdentityPreprocessor":
        return self

    def transform(self, signal: np.ndarray) -> np.ndarray:
        return signal

    def reconstruct_autoregressive(self, predicted_processed: np.ndarray) -> np.ndarray:
        return predicted_processed

    def reconstruct_onestep(self, predicted_processed: np.ndarray, aids: np.ndarray) -> np.ndarray:
        return predicted_processed + aids

    def prepare_for_inference(self, test_signal: np.ndarray, sequence_length: int) -> dict:
        X_test, y_test = create_sequences(test_signal, sequence_length)
        return {"X_test_model": X_test, "y_test_unscaled": y_test, "reconstruction_aids": np.zeros(len(X_test))}


class ScalingPreprocessor(BasePreprocessor):
    def __init__(self, feature_range: tuple[float, float] = (0, 1)):
        self.scaler = MinMaxScaler(feature_range=feature_range)

    def fit(self, train_signal: np.ndarray) -> "ScalingPreprocessor":
        self.scaler.fit(train_signal.reshape(-1, 1))
        return self

    def transform(self, signal: np.ndarray) -> np.ndarray:
        return self.scaler.transform(signal.reshape(-1, 1)).flatten()

    def reconstruct_autoregressive(self, predicted_processed: np.ndarray) -> np.ndarray:
        return self.scaler.inverse_transform(predicted_processed.reshape(-1, 1)).flatten()

    def reconstruct_onestep(self, predicted_processed: np.ndarray, aids: np.ndarray) -> np.ndarray:
        return self.scaler.inverse_transform(predicted_processed.reshape(-1, 1)).flatten()

    def prepare_for_inference(self, test_signal: np.ndarray, sequence_length: int) -> dict:
        scaled_test_signal = self.scaler.transform(test_signal.reshape(-1, 1)).flatten()
        X_test_scaled, _ = create_sequences(scaled_test_signal, sequence_length)
        _, y_test_unscaled = create_sequences(test_signal, sequence_length)
        return {
            "X_test_model": X_test_scaled,
            "y_test_unscaled": y_test_unscaled,
            "reconstruction_aids": np.zeros(len(X_test_scaled)),
        }


class ScalingAndDifferencingPreprocessor(BasePreprocessor):
    def __init__(self, feature_range: tuple[int, int] = (0, 1)):
        self.scaler = MinMaxScaler(feature_range=feature_range)
        self.last_train_value_scaled = None

    def fit(self, train_signal: np.ndarray) -> "ScalingAndDifferencingPreprocessor":
        self.scaler.fit(train_signal.reshape(-1, 1))
        scaled_train = self.scaler.transform(train_signal.reshape(-1, 1)).flatten()
        self.last_train_value_scaled = scaled_train[-1]
        return self

    def transform(self, signal: np.ndarray) -> np.ndarray:
        scaled_signal = self.scaler.transform(signal.reshape(-1, 1)).flatten()
        stationary_signal = scaled_signal[1:] - scaled_signal[:-1]
        return stationary_signal

    def reconstruct_autoregressive(self, predicted_processed: np.ndarray) -> np.ndarray:
        reconstructed_scaled = self.last_train_value_scaled + np.cumsum(predicted_processed)
        return self.scaler.inverse_transform(reconstructed_scaled.reshape(-1, 1)).flatten()

    def reconstruct_onestep(self, predicted_processed: np.ndarray, aids: np.ndarray) -> np.ndarray:
        predicted_values_scaled = aids + predicted_processed
        return self.scaler.inverse_transform(predicted_values_scaled.reshape(-1, 1)).flatten()

    def prepare_for_inference(self, test_signal: np.ndarray, sequence_length: int) -> dict:
        scaled_test_signal = self.scaler.transform(test_signal.reshape(-1, 1)).flatten()

        stationary_test_signal = scaled_test_signal[1:] - scaled_test_signal[:-1]

        X_test_diff, _ = create_sequences(stationary_test_signal, sequence_length)

        num_predictions = len(X_test_diff)

        reconstruction_aids = scaled_test_signal[sequence_length : sequence_length + num_predictions]

        y_test_unscaled = test_signal[sequence_length + 1 : sequence_length + 1 + num_predictions]

        return {
            "X_test_model": X_test_diff,
            "y_test_unscaled": y_test_unscaled,
            "reconstruction_aids": reconstruction_aids,
        }
