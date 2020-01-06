import json
from tqdm import tqdm
import numpy as np
from scipy.spatial.distance import euclidean

from search.document import Document
from search.question import Question
from search.tokenizer import Tokenizer
from search.embedder import Embedder

class SearchTable(object):
    def __init__(self):
        self.keywordIdx = {}
        self.clusterIdx = {}
        self.idIdx = {}
        self.pageNum = 0

        # processing modules
        self.tokenizer = Tokenizer(build='gpt')
        self.keyworder = None
        self.embedder = Embedder()

    def __str__(self):
        return f'<SearchTable | pageNum={self.pageNum}>'

    # DATA LOADING
    def add_document(self, document):
        ''' Adds single document to kewordIdx, clusterIdx, and idIdx '''
        self.add_keyword_doc(document)
        self.add_cluster_doc(document)
        self.add_id_doc(document)

    def add_keyword_doc(self, document):
        pass

    def add_cluster_doc(self, document):
        pass

    def add_id_doc(self, i, document):
        self.idIdx.update({i : document})

    # ABSTRACT TEXT MANIPULATION
    def make_question(self, id, text, span, asker):
        ''' Generates Question after encoding text '''
        textIds = self.tokenizer.string_to_ids(text)
        return Question(id, text, span, asker)

    # SQUAD LOADING
    def extract_squad_questions(self, questions, tokens):
        ''' Helper to find questions while building table '''
        for q in questions:
            if q['is_impossible']:
                yield self.make_question(q['id'], q['question'], None, 'squad')
                continue
            try:
                answerList = q['answers']
            except KeyError:
                answerList = q['plausible_answers']
                if (answerList == []):
                    continue

    def load_squad_file(self, path):
        ''' Loads squad file from path into various idxs '''
        with open(path, 'r') as squadFile:
            data = json.load(squadFile)['data']
            for category in tqdm(data, leave=False):
                title = category['title']
                for id, doc in enumerate(category['paragraphs']):
                    tokens = self.tokenizer.string_to_ids(doc['context'])



    def search(self, text):
        embedding = self.embedder.vectorize(text)
        print(f'LEN: {len(self.clusterIdx)}')
        scores = [(euclidean(e, embedding), t) for e, t in self.clusterIdx.values()]
        for _ in range(3):
            print(scores.pop(scores.index(max(scores)))[1], end=f'\n{"-"*80}\n')


    # DATA YIELDING
    def gen_qa_batch(self, n, maxLen):
        '''
        Genereates batch tensor of size n with form (n, maxLen, )
        '''
        pass
