'''
Attn encoder-decoder module for exploring attention methods for building
strong conditional dependence of decoder to avoid route language modeling.
'''

import torch
import numpy as np
from torch import nn
from tqdm import tqdm, trange
from termcolor import colored
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.cuda import is_available as gpu_available

device = torch.device("cuda" if gpu_available() else "cpu")

# class Lang(object):
#     def __init__(self, inVocab, outVocab):
#         assert len(set(inVocab)) == len(set(outVocab)), 'lens must be same'
#         self.inIdx = {i : word for i, word in enumerate(inVocab)}
#         self.inRev = {word : i for i, word in self.inIdx.items()}
#         self.outIdx = {i : word for i, word in enumerate(outVocab)}
#         self.outRev = {word : i for i, word in self.outIdx.items()}
#         self.voxLen = len(self.outIdx)
#         # build translation schema
#         self.transIdx = {inW : outW for inW, outW in zip(inVocab, outVocab)}
#         self.transRev = {outW : inW for inW, outW in self.transIdx.items()}
#
#     def gen_seqPair(self, inLen):
#         ''' gen pair of context, question tuple '''
#         inIds = np.random.randint(0, self.voxLen, size=inLen)
#         inWords = [self.inIdx[id] for id in inIds]
#         outWords = [self.transIdx[w] for w in inWords]
#         outIds = [self.outRev[w] for w in outWords]
#         inTensor = torch.tensor(inIds, dtype=torch.long,
#                                 device=device).view(-1, 1)
#         outTensor = torch.tensor(outIds, dtype=torch.long,
#                                 device=device).view(-1, 1)
#         return (inTensor, outTensor)
#
#     def str_seq(self, seq, n):
#         if n==0:
#             return ' '.join((self.inIdx[id] for id in seq))
#         elif n==1:
#             return ' '.join((self.outIdx[id] for id in seq))
#         else:
#             raise ValueError('n must be in {0, 1}')
#
#
# inVocab = ['hi', 'you', 'are', 'cool']
# outVocab = ['hola', 'tu', 'es', 'coolo']
# x = Lang(inVocab, outVocab)
# print(x.gen_seqPair(10))

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=100):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)
