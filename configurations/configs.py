from dataclasses import dataclass, field

import numpy as np
from torch import nn


@dataclass
class NoiseConfig:
    """Configuration for a noise to be applied to a signal.

    Attributes:
        name (str): Unique name of the noise configuration.
        type (str): Type of noise (e.g., 'gaussian', 'uniform').
        n_params (dict): Additional parameters for the noise generation.
            Defaults to an empty dictionary.
    """

    name: str
    type: str
    n_params: dict = field(default_factory=lambda: {})


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment.

    Attributes:
        exp_name (str): Name of the experiment.
        model (str): Name of the model to train (e.g., 'lstm', 'gru').
        model_params (dict): Parameters for the model initialization.
        optimizer (str): Name of the optimizer (e.g., 'adam', 'sgd').
        optimizer_params (dict): Parameters for the optimizer initialization.
        criterion (nnn.Module): Criterion
        datasets (List[str]): List of dataset names to use for training.
    """

    exp_name: str
    model: str
    model_params: dict
    optimizer: str
    optimizer_params: dict
    criterion: nn.Module
    datasets: list[str]
    politic: str = "standard"
    politic_params: dict = field(default_factory=dict)
    n_runs: int = 5


@dataclass
class SessionConfig:
    """Configuration for a session of experiments.

    A session can contain multiple experiments with shared common parameters
    and a signal used for testing.

    Attributes:
        session_name (str): Name of the session.
        common_params (dict): Parameters common to all experiments in the session
            (e.g., batch_size, sequence_length, epochs).
        experiments (List[ExperimentConfig]): List of experiments included in this session.
        test_signal (np.ndarray): Signal used for testing the trained models.
    """

    session_name: str
    common_params: dict
    experiments: list[ExperimentConfig]
    test_signal: np.ndarray


@dataclass
class DatasetConfig:
    """
    Attributes:
        name (str)
        signal (np.ndarray)
        metadata (Dict[str, Any])
    """

    name: str
    signal: np.ndarray
    metadata: dict = field(default_factory=dict)
