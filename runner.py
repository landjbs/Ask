from search.searchtable import SearchTable

x = SearchTable()


x.load_squad_file('data/inData/dev-v2.0.json')

print(x.idIdx)

x.load_squad_file('data/inData/dev-v2.0.json')

print(x.idIdx)
