from dataclasses import dataclass, field
from typing import Any


@dataclass
class TuningConfig:
    study_name: str
    n_trials: int

    model: str
    model_params: dict[str, Any]

    optimizer: str
    optimizer_params_grid: dict[str, Any]

    criterion_class: Any
    criterion_params_grid: dict[str, Any]

    data_generation_config: dict[str, Any] = field(default_factory=dict)

    metric_direction: str = "minimize"
