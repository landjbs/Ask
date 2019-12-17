'''
To read datasets into structs
'''

# import torch
# from transformers import GPT2LMHeadModel, GPT2Tokenizer
# t = GPT2Tokenizer.from_pretrained('gpt2')
# m = GPT2LMHeadModel.from_pretrained('gpt2')
# m.eval()
# print(dir(t))

from structs import SearchTable
from longshot import LongShot

SQUAD_PATH = 'data/inData/dev-v2.0.json'

# x = SearchTable(SQUAD_PATH)
x = SearchTable(loadPath='SearchTable')
# x.save('SearchTable')
z = LongShot(x)
z.train(100)
