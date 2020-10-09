import torch
from torch import nn

rnn = nn.RNN(100, 10)
print(rnn._parameters.key())