import torch
import numpy as np
from torch import nn
from tqdm import tqdm, trange
from termcolor import colored
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.cuda import is_available as gpu_available

from answer import 
