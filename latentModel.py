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

from transformers import GPT2LMHeadModel, GPT2Tokenizer

from structs import SearchTable

x = SearchTable(loadPath='SearchTable')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = GPT2LMHeadModel.from_pretrained('gpt2')  # or any other checkpoint
word_embeddings = model.transformer.wte.weight  # Word Token Embeddings
position_embeddings = model.transformer.wpe.weight  # Word Position Embeddings 
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
text_index = tokenizer.encode('man',add_prefix_space=True)
vector = model.transformer.wte.weight[text_index,:]
print(vector)
