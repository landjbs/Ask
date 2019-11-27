'''
To read datasets into structs
'''

from structs import SearchTable


x = SearchTable('data/inData/dev-v2.0.json')

print(x.char_tokenize('hi!'))

# for z in x.iter_docs():
#     for q in z.iter_questions():
#         print(q)
