import torch

from models.simple_lstm import SimpleLSTM

MODELS_MAP = {"lstm": SimpleLSTM}

OPTIMIZERS_MAP = {"adam": torch.optim.Adam, "sgd": torch.optim.SGD, "rmsprop": torch.optim.RMSprop}
