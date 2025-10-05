import colorednoise
import numpy as np


def _normalize_noise(noise: np.ndarray, mean: float, variance: float) -> np.ndarray:
    noise = noise - np.mean(noise)
    current_std = np.std(noise)
    if current_std > 0:
        noise = noise / current_std
    desired_std = np.sqrt(variance)
    return mean + desired_std * noise


def generate_noise(n_samples: int, color: str = "white", mean: float = 0, variance: float = 1) -> np.ndarray:
    beta_map = {"white": 0, "pink": 1, "red": 2, "brownian": 2, "blue": -1, "violet": -2}

    beta = beta_map.get(color.lower())
    if beta is None:
        raise ValueError(f"Неизвестный цвет шума: {color}. Доступные цвета: {list(beta_map.keys())}")

    noise = colorednoise.powerlaw_psd_gaussian(beta, n_samples)

    return _normalize_noise(noise, mean, variance)


def mix_noises(n_samples: int, mix_noises_config: list[dict], weights: list | np.ndarray) -> np.ndarray:
    if len(mix_noises_config) != len(weights):
        raise ValueError("Количество конфигураций шума должно совпадать c количеством весов.")

    weights = np.array(weights, dtype=float)
    weights /= np.sum(weights)

    mixed_noise = np.zeros(n_samples)

    for config, weight in zip(mix_noises_config, weights):
        noise_component = generate_noise(
            n_samples=n_samples,
            color=config.get("color", "white"),
            mean=config.get("mean", 0),
            variance=config.get("variance", 1),
        )
        mixed_noise += weight * noise_component

    return mixed_noise
