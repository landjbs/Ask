import json
import jsonlines
from tqdm import tqdm
import numpy as np
from scipy.spatial.distance import euclidean

from search.document import Document
from search.question import Question
from search.tokenizer import Tokenizer
# from search.embedder import Embedder

class SearchTable(object):
    def __init__(self):
        self.keywordIdx = {}
        self.clusterIdx = {}
        self.idIdx = {}
        self.pageNum = 0

        # processing modules
        self.tokenizer = Tokenizer(build='gpt')
        self.keyworder = None
        self.embedder = None

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
        if isinstance(text, str):
            text = self.tokenizer.string_to_ids(text)
        return Question(id, text, span, asker)

    def make_document(self, id, questions, text, path):
        if isinstance(text, str):
            text = self.tokenizer.string_to_ids(text)
        vec = None
        tokens = None
        return Document(id, tokens, questions, vec, text, path)

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
            # focuses only on the first answer of answer list
            answer = answerList[0]
            answerTokens = self.tokenizer.string_to_ids(answer['text'])
            answerStart = answerTokens[0]
            answerLen = len(answerTokens)
            span = None
            for loc, word in enumerate(tokens):
                if (word==answerStart):
                    if ((tokens[loc:(loc+answerLen)]==answerTokens)
                        and (answerLen>1)):
                        span = (loc, loc+answerLen)
                        break
            yield self.make_question(q['id'], q['question'], span, 'squad')

    def load_squad_document(self, paragraphs, inc):
        ''' Helper to load documents from list of paragraphs '''
        for dId, doc in enumerate(paragraphs):
            tokens = self.tokenizer.string_to_ids(doc['context'])
            questions = {qId : qObj for qId, qObj
                         in enumerate(self.extract_squad_questions(doc['qas'],
                                                                   tokens))}
            yield self.make_document((dId+inc), questions, tokens, None)

    def load_squad_file(self, path):
        ''' Loads squad file from path into various idxs '''
        with open(path, 'r') as squadFile:
            data = json.load(squadFile)['data']
            inc = 0 if (self.idIdx=={}) else max(self.idIdx)
            for category in tqdm(data, leave=False):
                title = category['title']
                paragraphs = category['paragraphs']
                docs = {(id+inc) : doc for id, doc
                        in enumerate(self.load_squad_document(paragraphs, inc))}
            if (inc==0):
                self.idIdx = docs
            else:
                self.idIdx.update(docs)
            return True

    def load_nq_file(self, path):
        with jsonlines(path, 'r') as nqReader:
            for obj in nqReader:
                print(obj)


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
