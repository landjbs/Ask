'''
To read datasets into structs
'''

from structs import SearchTable


x = SearchTable()
x.build('data/inData/train-v2.0.json')
x.save('SearchTable')
