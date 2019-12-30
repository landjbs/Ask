'''
Base models to be added or adapted for higher level Ask and Answer models.
'''

import torch
import numpy as np
from torch import nn
from tqdm import tqdm, trange
from termcolor import colored
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.cuda import is_available as gpu_available


class Encoder(nn.Module):
    '''
    Encodes text ids to generate attention matrix to concatenate between
    encoded question and context and hidden state for dynamic decoder and
    Ask approximation. Hidden state of context encoding is initialized with
    output hidden state of question encoding.
    '''
    def __init__(self, inDim, hiddenDim, layerNum):
        super(Encoder, self).__init__()
        # attributes
        self.inDim = inDim
        self.hiddenDim = hiddenDim
        self.layerNum = layerNum
        # sentinel vector is random noise to allow attenuation to impossibles
        self.sentinel = nn.Parameter(torch.rand(hiddenDim))
        # layers
        self.embedding = nn.Embedding(inDim, hiddenDim)
        self.rnn = nn.GRU(input_size=hiddenDim,
                          hidden_size=hiddenDim,
                          num_layers=layerNum,
                          bidirectional=False)

    def init_hidden(self, device):
        return torch.zeros(self.layerNum, 1, self.hiddenDim, device=device)

    def forward(self, inputId, hidden):
        out = self.embedding(inputId).view(1, 1, -1)
        out, hidden = self.rnn(out, hidden)
        return out, hidden
