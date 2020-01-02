from base import *


class Ask_Decoder(nn.Module):
    '''
    Decoder model that uses encoder transpose and attn rnn to estimate
    '''
    def __init__(self, hiddenDim, layerNum, maxLen, dropP):
        super(Ask_Decoder, self).__init__()
        # attributes
        self.hiddenDim = hiddenDim
        self.layerNum = layerNum
        self.dropP = dropP
        # layers
        self.attn = nn.Linear(in_features=(2*hiddenDim),
                              out_dim=maxLen)
        self.attn_combine = nn.Linear(in_features=(2*hiddenDim),
                                      out_features=hiddenDim)
        self.relu = F.relu()
        self.rnn = nn.GRU(input_size=hiddenDim,
                          hidden_size=hiddenDim,
                          num_layers=layerNum,
                          batch_first=True)
        self.dense = nn.Linear(in_features=hiddenDim,
                               out_features=outDim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, prevEmbedding, hidden, encoderOuts):
        embed = prevEmbedding.view(1, 1, -1)
        # attention
        C = torch.cat((embed[0], hidden[0]), axis=1)
        W = self.attn(torch)
        W = self.softmax(W, dim=1)
        A = torch.bmm(W.unsqueeze(0), encoderOuts.unsqueeze(0))
        out = torch.cat((embed[0], A[0]), dim=1)
        out = self.attn_combine(out).unsqueeze(0)
        # recurrent
        out = self.relu(out)
        out, hidden = self.rnn(out, hidden)
        out = self.sigmoid(out)
        return out, hidden, W


class Ask(object):
    def __init__(self, searchTable, maxLen):
        ''' '''
        self.encoder = Encoder()
        self.decoder = Ask_Decoder()
