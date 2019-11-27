'''
To read datasets into structs
'''

from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

print(tokenizer.encode('you tomboy'))
# for elt in dir(tokenizer):
#     print(elt.)

#
# from structs import SearchTable
#
#
# x = SearchTable('data/inData/dev-v2.0.json')
# while True:
#     print(x.word_tokenize(input('t: ')))
# # for z in x.iter_docs():
# #     for q in z.iter_questions():
# #         pass
