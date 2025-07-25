import torch.nn as nn
from torch.optim import Adam, SGD

optimizers = {
    'Adam': Adam,
   'SGD': SGD
}

activations = {
    'Tanh': nn.Tanh,
    'ReLU': nn.ReLU,
    'LeakyReLU': nn.LeakyReLU,
    'Sigmoid': nn.Sigmoid,    
    "LogSigmoid": nn.LogSigmoid,
    "Softmax": nn.Softmax,
    "LogSoftmax": nn.LogSoftmax
}