'''
Answer model uses GRU over GPT2 embeddings followed by convo and dense layers
to employ span selection.

Uses multi-layer Gated Recurrent Unit to separately encode quesiton
and context. Dynamic Co-Attention Mechanism allows bidirectional attentuation
to elements of Q and C for each decode step.
Highway Maxout Ensemble Model with random initializations allows colaboration
to raise final performance. Mean final hidden state of encoder ensemble across
novel question can be estimated by encoder in Ask longshot question generation
to expediate training.
https://arxiv.org/pdf/1611.01604.pdf
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

import utils as u

ZERO_BOOSTER = 0.000000001

device = torch.device("cuda" if gpu_available() else "cpu")


class Encoder(nn.Module):
    '''
    Encodes just the question to generate hidden state for ASK approximation
    and attention matrix to concat with attention matrix of C_Encoder
    '''
    def __init__(self, inDim, hiddenDim, layerNum=1, dropoutPercent=0.1):
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
                          batch_first=True,
                          dropout=dropoutPercent,
                          bidirectional=False)
        self.dropout = nn.Dropout(p=dropoutPercent)
        # self.nonLinearity = nn.ReLU()

    def init_hidden(self, device):
        return torch.zeros(1, 1, self.hiddenDim, device=device)

    def forward(self, seq, mask):
        # take sum of the mask matrix
        lens = torch.sum(mask, 1)
        print(lens)
        valSort, iSort = torch.sort(lens, dim=0, descending=True)
        _, iAscending = torch.sort(iSort, dim=0, descending=False)
        seq_ = torch.index_select(input=seq, dim=0, index=iSort)
        embed = self.embedding(seq_)
        # use util to pack embeddings for batch encoding
        packedSeq = pack_padded_sequence(embed, valSort, batch_first=True)
        out, _ = self.rnn(packedSeq)
        e, _ = pad_packed_sequence(out, batch_first=True)
        e = e.contiguous()
        e = torch.index_select(e, dim=0, index=iAscending)
        e = self.dropout(e)
        b, _ = list(mask.size())
        # add the sentinel vector
        sentinelExp = self.sentinel.squeeze().expand(b, self.hiddenDim).unsqueeze(1).contiguous()
        lens = lens.unsqueeze(1).expand(b, self.hiddenDim).unsqueeze(1)
        sentinelZero = torch.zeros(b, self.hiddenDim).unsqueeze(1)
        e = torch.cat((e, sentinelZero), dim=1)
        e = e.scatter_(1, lens, sentinelExp)
        return e


class Dense(nn.Module):
    '''
    Non-linearity to allow unique encoding for questions and context.
    '''
    def __init__(self, hiddenDim):
        super(Dense, self).__init__()
        self.dense = nn.Linear(in_features=hiddenDim, out_features=hiddenDim)
        self.tan = nn.Tanh()

    def forward(self, encoderOut):
        return self.tan(self.dense(encoderOut))


class Encoder(nn.Module):
    '''
    Encodes text ids to generate attention matrix to concatenate between
    encoded question and context and hidden state for dynamic decoder and
    Ask approximation. Hidden state of context encoding is initialized with
    output hidden state of question encoding.
    '''
    def __init__(self, nonLinearity, inDim, hiddenDim, layerNum):
        assert isinstance(nonLinearity, Dense), 'nonLinearity should be Dense.'
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
        self.nonLinearity = nonLinearity

    def init_hidden(self, device):
        return torch.zeros(self.layerNum, 1, self.hiddenDim, device=device)

    def forward(self, inputId, hidden):
        out = self.embedding(inputId).view(1, 1, -1)
        out, hidden = self.rnn(out, hidden)
        out = self.nonLinearity(out)
        return out, hidden


d = Dense(100)
e = Encoder(d, 10, 100, 2)

h = e.init_hidden(device)

encoderOptim = torch.optim.Adam(e.parameters(), lr=0.0005)
c = torch.zeros(100, dtype=torch.float)

for _ in range(100):
    t = torch.tensor([1,2,3])
    encoderOptim.zero_grad()
    loss = 0
    for i in t:
        o, h = e(i, h)
        loss += torch.log(torch.dist(o.float(), c)) ** 2
        print(loss)
        print(o)
    loss.backward()
    encoderOptim.step()

# class Q_Decoder(nn.Module):
#     def __init__(self, hiddenDim, maxLen, dropoutPercent):
#         super(Q_Decoder, self).__init__()
#         # attributes
#         self.hiddenDim = hiddenDim
#         self.maxLen = maxLen
#         self.dropoutPercent = dropoutPercent
#         # layers
#         self.embedding = nn.Embedding(outDim, hiddenDim)
#         self.attn = nn.Linear(in_features=(2*hiddenDim),
#                               out_features=maxLen)
#         self.attn_combine = nn.Linear(in_features=(2*hiddenDim),
#                                       out_features=hiddenDim)
#         self.dropout = nn.Dropout(p=dropoutPercent)
#         self.rnn = nn.GRU(input_size=hiddenDim,
#                           hidden_size=hiddenDim)
#         self.out = nn.Linear(in_features=hiddenDim, out_features=1)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, inputId, hidden, encoderOuts):
#         # embedding
#         embed = self.embedding(inputId).view(1, 1, -1)
#         embed = self.dropout()
#         # attention
#         attnWeights = self.attn(torch.cat((embed[0], hidden[0]), axis=1))
#         attnWeights = F.softmax(attnWeights, dim=1)
#         attnApplied = torch.bmm(attnWeights.unsqueeze(0),
#                                 encoderOuts.unsqueeze(0))
#         out = torch.cat((embed[0], attnApplied[0]), 1)
#         out = self.attn_combine(out).unsqueeze(0)
#         # recurrent
#         out = F.relu(out)
#         out, hidden = self.rnn(out)
#         out = self.sigmoid(out)
#         return out, hidden, attnWeights
#
#
# class Answer_Model(object):
#     def __init__(self, searchTable, qMax, cMax, hiddenDim, lr):
#         assert searchTable.initialized, 'SearchTable must be initialized.'
#         # attributes
#         self.inDim = searchTable.wordEmbeddingSize
#         self.hiddenDim = hiddenDim
#         self.searchTable = searchTable
#         self.qMax = qMax
#         self.cMax = cMax
#         self.startId = searchTable.word_encode(searchTable.startToken)
#         self.device = torch.device("cuda" if gpu_available() else "cpu")
#         # models
#         self.qEncoder = self.qEncoder(self.inDim, self.hiddenDim, layerNum=1)
#         self.cEncoder = self.cEncoder(self.inDim, self.hiddenDim, layerNum=1)
#         self.cDecoder = self.cDecoder(self.hiddenDim, self.outDim,
#                                       self.cMax, dropout=0.1)
#         self.qEncoderOptim = torch.optim.Adam(self.qEncoder.params(), lr=lr)
#         self.cEncoderOptim = torch.optim.Adam(self.cEncoder.params(), lr=lr)
#         self.cDecoderOptim = torch.optim.Adam(self.cDecoder.params(), lr=lr)
#         self.criterion = torch.nn.BCELoss()
#
#     def encode_question(self, qIds):
#         ''' Uses qEncoder to encode qIds into tuple (hidden, attn) '''
#         hidden = self.qEncoder.init_hidden(self.device)
#         outs = torch.zeros(self.qMax, self.hiddenDim, device=self.device)
#         for step, id in enumerate(qIds):
#             out, hidden = self.qEncoder(id, hidden)
#             outs[step] = out[0, 0]
#         return (hidden, outs)
#
#     def encode_context(self, cIds, hidden):
#         '''
#         Uses cEncoder and qEncoder hidden out to encode cIds into bidirectional
#         tuple of (hidden, attn)
#         '''
#         outs = torch.zeros(self.cMax, self.hiddenDim, device=self.device)
#         for step, id in enumerate(cIds):
#             out, hidden = self.cEncoder(id, hidden)
#             outs[step] = out[0, 0]
#         return (hidden, outs)
#
#     def decode(self, qIds, cIds, hidden):
#         ''' Decodes with no training or loss '''
#         hidden, qOuts = self.encode_question(qIds)
#         hidden, cOuts = self.encode_context(cIds, hidden)
#         eOuts = torch.cat((qOuts, cOuts), axis=1)
#         dIn = self.startId
#         decoded = torch.zeros(len(self.cIds))
#         for step in range(len(cIds)):
#             dOut, hidden = self.cDecoder(dIn, hidden, eOuts)
#             _, predLoc = dOut[0].topk(1)
#             dIn = predLoc.squeeze().detach()
#             decoded[step] = dIn
#         return decoded
#
#     def train_step(self, qIds, cIds, targets, force):
#         '''
#         Trains Answer_Model on text-question pair using binary target vector
#         to prop loss through qEncoder, cEncoder, cDecoder.
#         Args:
#             qIds:       Tensor of question ids.
#             cIds:       Tensor of context ids.
#             targets:    Binary tensor of targets with lenth equal to cIds.
#             force:      Whether or not to using teacher forcing.
#         Returns:
#             Loss of model at currents step.
#         '''
#         # clear optimizers
#         self.qEncoderOptim.zero_grad()
#         self.cEncoderOptim.zero_grad()
#         self.cDecoderOptim.zero_grad()
#         loss, numCorrect = 0, 0
#         # get encodings
#         hidden, qOuts = self.encode_question(qIds)
#         hidden, cOuts = self.encode_context(cIds, hidden)
#         # concatenate encoder outs
#         eOuts = torch.cat((qOuts, cOuts), axis=1)
#         dIn = self.startId
#         if force:
#             for step in range(len(targets)):
#                 dOut, hidden = self.cDecoder(dIn, hidden, eOuts)
#                 _, predLoc = dOut[0].topk(1)
#                 dIn = targets[step]
#                 loss += self.criterion(dOut, targets[step])
#         else:
#             for step in range(len(targets)):
#                 dOut, hidden = self.cDecoder(dIn, hidden, eOuts)
#                 _, predLoc = dOut[0].topk(1)
#                 dIn = predLoc.squeeze().detach()
#                 loss += self.criterion(dOut, targets[step])
#         loss.backward()
#         self.qEncoderOptim.step()
#         self.cEncoderOptim.step()
#         self.cDecoderOptim.step()
#         return (loss.item() / len(targets))
#
#     def train(self, epochs):
#         ''' Number of epochs for which to train on searchTable '''
#         pass
