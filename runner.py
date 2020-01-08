from search.searchtable import SearchTable

x = SearchTable()


x.load_squad_file('data/inData/dev-v2.0.json')

print(x.idIdx)

print(x.idIdx[0].pretty_print(0, x.tokenizer))
