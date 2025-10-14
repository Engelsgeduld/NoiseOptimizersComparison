from abc import ABC, abstractmethod

import numpy as np
from sequences_creator import create_sequences
from sklearn.preprocessing import MinMaxScaler


class BasePreprocessor(ABC):
    @abstractmethod
    def fit_transform(self, train_signal: np.ndarray) -> np.ndarray:
        pass

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
    def fit_transform(self, train_signal: np.ndarray) -> np.ndarray:
        return train_signal

    def reconstruct_autoregressive(self, predicted_processed: np.ndarray) -> np.ndarray:
        return predicted_processed

    def reconstruct_onestep(self, predicted_processed: np.ndarray, aids: np.ndarray) -> np.ndarray:
        return predicted_processed + aids

    def prepare_for_inference(self, test_signal: np.ndarray, sequence_length: int) -> dict:
        X_test, y_test = create_sequences(test_signal, sequence_length)
        return {"X_test_model": X_test, "y_test_unscaled": y_test, "reconstruction_aids": np.zeros(len(X_test))}


class ScalingAndDifferencingPreprocessor(BasePreprocessor):
    def __init__(self, feature_range: tuple[int, int]=(0, 1)):
        self.scaler = MinMaxScaler(feature_range=feature_range)
        self.last_train_value_scaled = None

    def fit_transform(self, train_signal: np.ndarray) -> np.ndarray:
        scaled_signal = self.scaler.fit_transform(train_signal.reshape(-1, 1)).flatten()
        self.last_train_value_scaled = scaled_signal[-1]
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
        X_test_scaled, _ = create_sequences(scaled_test_signal, sequence_length)

        stationary_test_signal = scaled_test_signal[1:] - scaled_test_signal[:-1]
        X_test_diff, _ = create_sequences(stationary_test_signal, sequence_length)

        _, y_test_unscaled = create_sequences(test_signal, sequence_length)

        return {
            "X_test_model": X_test_diff,
            "y_test_unscaled": y_test_unscaled,
            "reconstruction_aids": X_test_scaled[:, -1],
        }
