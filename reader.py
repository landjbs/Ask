'''
To read datasets into structs
'''

from structs import SearchTable


x = SearchTable()
x.build('data/inData/train-v2.0.json')
print(x)
x.save('test')
del x

x = SearchTable('test')
print(x)
