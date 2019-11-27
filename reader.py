'''
To read datasets into structs
'''

from transformers import GPT2LMHeadModel, GPT2Tokenizer
t = GPT2Tokenizer.from_pretrained('gpt2')
print(t.encode('you are my best', add_special_tokens=True))


# from structs import SearchTable
#
#
# x = SearchTable('data/inData/dev-v2.0.json')
# while True:
    # print(x.word_tokenize(input('t: ')))
