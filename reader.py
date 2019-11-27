'''
To read datasets into structs
'''

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
t = GPT2Tokenizer.from_pretrained('gpt2')
m = GPT2LMHeadModel.from_pretrained('gpt2')
m.eval()

e = t.encode('you are my best', add_special_tokens=False)
print([len(m.transformer.wte.weight[w]) for w in e])

# from structs import SearchTable
#
#
# x = SearchTable('data/inData/dev-v2.0.json')
# while True:
    # print(x.word_tokenize(input('t: ')))
