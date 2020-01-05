from search.searchtable import SearchTable

x = SearchTable()


x.load_squad_file('data/inData/dev-v2.0.json')


while True:
    t = input('t: ')
    x.search(t)
