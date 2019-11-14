'''
To read datasets into structs
'''

import json
from structs import Document, SearchTable

with open('data/inData/dev-v2.0.json', 'r') as squadFile:
    data = json.load(squadFile)
    for category in data['data']:
        title = category['title']
        for document in category['paragraphs']:
            text = document['context']
            questions = document['qas']
