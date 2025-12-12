import torch

from models.scheduled_lstm import ScheduledSamplingLSTM
from models.simple_lstm import SimpleLSTM

MODELS_MAP = {"lstm": SimpleLSTM, "shelduled_lstm": ScheduledSamplingLSTM}

OPTIMIZERS_MAP = {"adam": torch.optim.Adam, "sgd": torch.optim.SGD, "rmsprop": torch.optim.RMSprop}
