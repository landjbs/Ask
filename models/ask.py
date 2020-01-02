from base import *

class Ask_Decoder(nn.Module):
    '''
    Decoder model that uses encoder transpose and attn rnn to estimate
    '''
    def __init__(self, hiddenDim, layerNum):
        super(Ask_Decoder, self).__init__()
        # attributes
        self.hiddenDim = hiddenDim
        self.layerNum = layerNum
        # layers
        self.rnn = nn.GRU(input_size=hiddenDim)


class Ask(object):
    def __init__(self, searchTable, maxLen):
        ''' '''
        self.encoder = Encoder()
