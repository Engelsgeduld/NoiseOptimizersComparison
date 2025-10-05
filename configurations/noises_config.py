from dataclasses import dataclass, field


@dataclass
class NoiseConfig:
    name: str
    type: str
    n_params: dict = field(default_factory=lambda: {})
