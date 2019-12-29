'''
Answer model uses GRU over GPT2 embeddings followed by convo and dense layers
to employ span selection.
'''


import torch
import numpy as np
from torch import nn
from tqdm import tqdm, trange
from termcolor import colored
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.cuda import is_available as gpu_available

import utils as u

ZERO_BOOSTER = 0.000000001


class Q_Encoder(nn.Module):
    '''
    Encodes just the question to generate hidden state for ASK approximation
    and attention matrix to concat with attention matrix of C_Encoder
    '''
    def __init__(self, inDim, hiddenDim, layerNum):
        super(Q_Encoder, self).__init__()
        # attributes
        self.inDim = inDim
        self.hiddenDim = hiddenDim
        self.layerNum = layerNum
        # sentinel vector is random noise to allow attenuation to non-answers
        self.sentinel = nn.Parameter(torch.rand(hiddenDim))
        # layers
        self.embedding = nn.Embedding(inDim, hiddenDim)
        self.rnn = nn.GRU(input_size=hiddenDim,
                          hidden_size=hiddenDim,
                          num_layers=layerNum,
                          bidirectional=False)

    def init_hidden(self, device):
        return torch.zeros(1, 1, self.hiddenDim, device=device)

    def forward(self, inputId, hidden):
        out = self.embedding(inputId).view(-1, 1, 1)
        out, hidden = self.rnn(out, hidden)
        return out, hidden


class C_Encoder(nn.Module):
    '''
    Encodes just the context to generate hidden state for Q_Decoder and
    attention matrxi to concat with attention matrix of Q_Encoder. Hidden
    state is initialized with output hidden state of Q_Enocder.
    Is bidirectional.
    '''
    def __init__(self, inDim, hiddenDim, layerNum):
        super(C_Encoder, self).__init__()
        # attributes
        self.inDim = inDim
        self.hiddenDim = hiddenDim
        self.layerNum = layerNum
        # layers
        self.embedding = nn.Embedding(inDim, hiddenDim)
        self.rnn = nn.GRU(input_size=hiddenDim,
                          hidden_size=hiddenDim,
                          num_layers=layerNum,
                          bidirectional=True)

    def forward(self, inputId, hidden):
        out = self.embedding(inputId)
        out, hidden = self.rnn(out, hidden)
        return out, hidden


class Q_Decoder(nn.Module):
    def __init__(self, hiddenDim, maxLen, dropoutPercent):
        super(Q_Decoder, self).__init__()
        # attributes
        self.hiddenDim = hiddenDim
        self.maxLen = maxLen
        self.dropoutPercent = dropoutPercent
        # layers
        self.embedding = nn.Embedding(outDim, hiddenDim)
        self.attn = nn.Linear(in_features=(2*hiddenDim),
                              out_features=maxLen)
        self.attn_combine = nn.Linear(in_features=(2*hiddenDim),
                                      out_features=hiddenDim)
        self.dropout = nn.Dropout(p=dropoutPercent)
        self.rnn = nn.GRU(input_size=hiddenDim,
                          hidden_size=hiddenDim)
        self.out = nn.Linear(in_features=hiddenDim, out_features=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputId, hidden, encoderOuts):
        # embedding
        embed = self.embedding(inputId).view(1, 1, -1)
        embed = self.dropout()
        # attention
        attnWeights = self.attn(torch.cat((embed[0], hidden[0]), axis=1))
        attnWeights = F.softmax(attnWeights, dim=1)
        attnApplied = torch.bmm(attnWeights.unsqueeze(0),
                                encoderOuts.unsqueeze(0))
        out = torch.cat((embed[0], attnApplied[0]), 1)
        out = self.attn_combine(out).unsqueeze(0)
        # recurrent
        out = F.relu(out)
        out, hidden = self.rnn(out)
        out = self.sigmoid(out)
        return out, hidden, attnWeights


class Answer_Model(object):
    def __init__(self, searchTable, qMax, cMax, hiddenDim, lr):
        assert searchTable.initialized, 'SearchTable must be initialized.'
        # attributes
        self.inDim = searchTable.wordEmbeddingSize
        self.hiddenDim = hiddenDim
        self.searchTable = searchTable
        self.qMax = qMax
        self.cMax = cMax
        self.startId = searchTable.word_encode(searchTable.startToken)
        self.device = torch.device("cuda" if gpu_available() else "cpu")
        # models
        self.qEncoder = self.qEncoder(self.inDim, self.hiddenDim, layerNum=1)
        self.cEncoder = self.cEncoder(self.inDim, self.hiddenDim, layerNum=1)
        self.cDecoder = self.cDecoder(self.hiddenDim, self.outDim,
                                      self.cMax, dropout=0.1)
        self.qEncoderOptim = torch.optim.Adam(self.qEncoder.params(), lr=lr)
        self.cEncoderOptim = torch.optim.Adam(self.cEncoder.params(), lr=lr)
        self.cDecoderOptim = torch.optim.Adam(self.cDecoder.params(), lr=lr)
        self.criterion = torch.nn.BCELoss()

    def encode_question(self, qIds):
        ''' Uses qEncoder to encode qIds into tuple (hidden, attn) '''
        hidden = self.qEncoder.init_hidden(self.device)
        outs = torch.zeros(self.qMax, self.hiddenDim, device=self.device)
        for step, id in enumerate(qIds):
            out, hidden = self.qEncoder(id, hidden)
            outs[step] = out[0, 0]
        return (hidden, outs)

    def encode_context(self, cIds, hidden):
        '''
        Uses cEncoder and qEncoder hidden out to encode cIds into bidirectional
        tuple of (hidden, attn)
        '''
        outs = torch.zeros(self.cMax, self.hiddenDim, device=self.device)
        for step, id in enumerate(cIds):
            out, hidden = self.cEncoder(id, hidden)
            outs[step] = out[0, 0]
        return (hidden, outs)

    def decode(self, qIds, cIds, hidden):
        ''' Decodes with no training or loss '''
        hidden, qOuts = self.encode_question(qIds)
        hidden, cOuts = self.encode_context(cIds, hidden)
        eOuts = torch.cat((qOuts, cOuts), axis=1)
        dIn = self.startId
        decoded = torch.zeros(len(self.cIds))
        for step in range(len(cIds)):
            dOut, hidden = self.cDecoder(dIn, hidden, eOuts)
            _, predLoc = dOut[0].topk(1)
            dIn = predLoc.squeeze().detach()
            decoded[step] = dIn
        return decoded

    def train_step(self, qIds, cIds, targets, force):
        '''
        Trains Answer_Model on text-question pair using binary target vector
        to prop loss through qEncoder, cEncoder, cDecoder.
        Args:
            qIds:       Tensor of question ids.
            cIds:       Tensor of context ids.
            targets:    Binary tensor of targets with lenth equal to cIds.
            force:      Whether or not to using teacher forcing.
        Returns:
            Loss of model at currents step.
        '''
        # clear optimizers
        self.qEncoderOptim.zero_grad()
        self.cEncoderOptim.zero_grad()
        self.cDecoderOptim.zero_grad()
        loss, numCorrect = 0, 0
        # get encodings
        hidden, qOuts = self.encode_question(qIds)
        hidden, cOuts = self.encode_context(cIds, hidden)
        # concatenate encoder outs
        eOuts = torch.cat((qOuts, cOuts), axis=1)
        dIn = self.startId
        if force:
            for step in range(len(targets)):
                dOut, hidden = self.cDecoder(dIn, hidden, eOuts)
                _, predLoc = dOut[0].topk(1)
                dIn = targets[step]
                loss += self.criterion(dOut, targets[step])
        else:
            for step in range(len(targets)):
                dOut, hidden = self.cDecoder(dIn, hidden, eOuts)
                _, predLoc = dOut[0].topk(1)
                dIn = predLoc.squeeze().detach()
                loss += self.criterion(dOut, targets[step])
        loss.backward()
        self.qEncoderOptim.step()
        self.cEncoderOptim.step()
        self.cDecoderOptim.step()
        return (loss.item() / len(targets))

    def train(self, epochs):
        ''' Number of epochs for which to train on searchTable '''
        pass
