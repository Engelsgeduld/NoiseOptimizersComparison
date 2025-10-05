import numpy as np

from configurations.noises_config import NoiseConfig
from noises.utils.noise_generator import _normalize_noise, generate_noise, mix_noises


class NoiseApplier:
    def __init__(self, signal: np.ndarray) -> None:
        if not isinstance(signal, np.ndarray):
            signal = np.array(signal, dtype=float)
        self.signal = signal
        self.n_samples = len(signal)

    def _calculate_noise_variance_for_snr(self, snr_db: float) -> float:
        signal_power = np.mean(self.signal**2)
        snr_linear = 10 ** (snr_db / 10)
        noise_variance = signal_power / snr_linear
        return noise_variance

    def apply(self, noise_configurations: list[NoiseConfig]) -> dict:
        noisy_datasets = {}

        for config in noise_configurations:
            name = config.name
            noise_type = config.type
            params = config.n_params

            noise = np.zeros(self.n_samples)

            if noise_type == "mix":
                noise_components = params["noises"]
                weights = params["weights"]

                temp_mixed_noise = mix_noises(self.n_samples, noise_components, weights)

                if "snr_db" in params:
                    target_variance = self._calculate_noise_variance_for_snr(params["snr_db"])
                    noise = _normalize_noise(temp_mixed_noise, 0, target_variance)
                else:
                    noise = temp_mixed_noise

            else:
                if "snr_db" in params:
                    variance = self._calculate_noise_variance_for_snr(params["snr_db"])
                    mean = params.get("mean", 0)
                else:
                    variance = params.get("variance", 1)
                    mean = params.get("mean", 0)

                noise = generate_noise(self.n_samples, color=noise_type, mean=mean, variance=variance)

            noisy_datasets[name] = self.signal + noise

        return noisy_datasets
