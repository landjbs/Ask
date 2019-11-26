import json
from tqdm import tqdm
from termcolor import colored
from itertools import izip

import utils as u

class Question(object):
    ''' Single question about a doc stores '''
    def __init__(self, questionId, qText, aText, aSpan):
        self.questionId = questionId
        self.qText = qText
        self.aText = aText
        self.aSpan = aSpan

    def __str__(self):
        return f"<QuestionObj | '{self.qText}' : '{self.aText}'>"

class Document(object):
    ''' Single document from any QA dataset '''
    def __init__(self, docId, category, text, questionIdx):
        self.docId = docId
        self.category = category
        self.text = text
        self.questionIdx = questionIdx

    def __str__(self):
        return f'<DocumentObj | id={self.docId}, category={self.category}>'

    def get_questions(self):
        ''' Gets list of questions pertaining to document '''
        return self.questionIdx.values()


class SearchTable(object):
    ''' Wide column hashtable of Document objects for searching '''
    def __init__(self, squadPath=None, loadPath=None):
        self.initialized = False
        if loadPath:
            self.load(loadPath)
        elif squadPath:
            self.build(squadPath)
        else:
            self.categoryIdx = {}

    def save(self, savePath):
        assert self.initialized, 'SearchTable must be initialized before save.'
        u.safe_make_folder(savePath)
        u.save_obj(self.categoryIdx, f'{savePath}/categoryIdx')
        return True

    def load(self, loadPath):
        assert not self.initialized, 'Cannot load into initialized SearchTable.'
        u.path_exists(loadPath)
        self.categoryIdx = u.load_obj(f'{loadPath}/categoryIdx')
        self.initialized = True

    def __str__(self):
        outStr = 'SearchTableObj\n'
        for category, contents in self.categoryIdx.items():
            outStr += f'\t{category} : {len(contents)}\n'
        return outStr

    def __iter__(self):
        return [(category, contents) for category, contents
                in self.categoryIdx.items()]

    def iter_docs(self):
        ''' Iterate over documents, ignoring category '''
        for doc in izip(self.categoryIdx.values()):
            yield doc

    # TEXT MANIPULATION
    def _tokenize(self, text):
        ''' returns tokenized text; to improve '''
        return text.lower().split()

    # DATA ANALYSIS
    def _question_extractor(self, questions, textWords):
        ''' Helper to find question in table building '''
        for q in questions:
            if q['is_impossible']:
                continue
            try:
                answerList = q['answers']
            except KeyError:
                answerList = []
            if (answerList == []):
                answerList = q['plausible_answers']
                if (answerList == []):
                    continue
            # focuses only on the first answer of answer list
            answer = answerList[0]
            answerText = answer['text']
            answerWords = self._tokenize(answerText)
            answerStart = answerWords[0]
            answerLen = len(answerWords)
            span = None
            for loc, textWord in enumerate(textWords):
                if (textWord==answerStart):
                    if (textWords[loc:(loc+answerLen)] == answerWords):
                        span = (loc, loc+answerLen)
                        break
            if not span:
                continue
            yield Question(q['id'], q['question'], answerText, span)

    def _document_extractor(self, category):
        ''' Helper to find documents in table building '''
        title = category['title']
        for docId, document in enumerate(category['paragraphs']):
            text = document['context']
            textWords = self._tokenize(text)
            questions = document['qas']
            questionIdx = {i : qObj for i, qObj
                            in enumerate(self._question_extractor(questions,
                                                                  textWords))}
            yield Document(docId, title, text, questionIdx)

    def build(self, squadPath):
        ''' Builds SearchTable from squad file under squadPath'''
        print(colored('BUILDING SEARCH TABLE', 'yellow'))
        print(colored('Analyzing files:', 'red'))
        with open(squadPath, 'r') as squadFile:
            data = json.load(squadFile)['data']
            self.categoryIdx = {category['title'] :
                                list(self._document_extractor(category))
                                for category in tqdm(data)}
        print(colored('Indexing files:', 'red'))
        self.initialized = True
        print(colored('SEARCH TABLE BUILT', 'green'))
        return True
