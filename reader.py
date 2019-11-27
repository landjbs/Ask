'''
To read datasets into structs
'''

# import torch
# from transformers import GPT2LMHeadModel, GPT2Tokenizer
# t = GPT2Tokenizer.from_pretrained('gpt2')
# m = GPT2LMHeadModel.from_pretrained('gpt2')
# m.eval()

from structs import SearchTable

x = SearchTable('data/inData/dev-v2.0.json')
