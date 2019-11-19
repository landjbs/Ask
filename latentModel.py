'''
The latent model attempts to approximate the CLS vector embedding of questions
given their span and the text. Uses decoder RNN over GPT-2 token embeddings
with final binary dimension classifying token as element in or out of span.
Uses ReLu dense final layer on cell state of RNN followed by norming by mean
of GPT-2 embeddings (to accomodate negative values) to output embedding vector
approxmation of question relating to span.
'''

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from structs import SearchTable

x = SearchTable(loadPath='SearchTable')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AttnRNN(object):
    def __init__(self,)
