from base import *

class Ask_Decoder(nn.Module):
    '''
    Decoder model that uses encoder transpose and attn rnn to estimate
    '''
    def __init__(self):
        pass

class Ask(object):
    def __init__(self, searchTable, maxLen):
        ''' '''
        self.encoder = Encoder()
