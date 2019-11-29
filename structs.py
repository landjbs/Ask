import re
import json
import numpy as np
from tqdm import tqdm
from termcolor import colored
from itertools import chain
from transformers import GPT2LMHeadModel, GPT2Tokenizer

import utils as u

class Question(object):
    ''' Single question about a doc stores '''
    def __init__(self, questionId, qText, aSpan):
        self.questionId = questionId
        self.qText = qText
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

    def iter_questions(self):
        ''' Iterates over questions in doc, yielding tuple (text, span) '''
        for question in self.questionIdx.values():
            yield (question.qText, question.aSpan)


class SearchTable(object):
    ''' Wide column hashtable of Document objects for searching '''
    def __init__(self, squadPath=None, loadPath=None):
        self.initialized = False
        # load gpt2 models
        self.gptModel = GPT2LMHeadModel.from_pretrained('gpt2')
        self.gptTokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.gptModel.eval()
        # store dict for char embeddings
        charList = [c for c in 'abcdefghijklmnoqrstuvwxyz0123456789']
        self.startToken = '`'
        self.endToken = '#'
        charList += [self.startToken] + [self.endToken]
        charIdx = {c : i for i, c in enumerate(charList)}
        revCharIdx = {i : c for i, c in enumerate(charList)}
        self.char_to_id = lambda c : charIdx[c]
        self.id_to_char = lambda i : revCharIdx[i]
        self.charMatcher = re.compile("|".join(charList))
        # define global vars
        self.wordEmbeddingSize = 768
        self.charEmbeddingSize = len(charList)
        # determine whether to load data
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
        for categoryDocs in self.categoryIdx.values():
            for doc in categoryDocs:
                yield doc

    # TEXT MANIPULATION
    def word_tokenize(self, text):
        ''' Returns word-tokenized text using gptTokenizer '''
        return self.gptTokenizer._tokenize(text.lower().strip())

    def word_encode(self, tokenList):
        ''' Returns id-encoded list from word-tokens using gptTokenizer '''
        return self.gptTokenizer.convert_tokens_to_ids(tokenList)

    def word_embed(self, idList):
        ''' Returns list of embedding vectors for ids !USES LOOKUP: BAD! '''
        return [self.gptModel.transformer.wte.weight[id].detach().numpy()
                for id in idList]

    def char_tokenize(self, text):
        ''' Returns char-tokenized text using charMatcher '''
        return self.charMatcher.findall(text.lower().strip())

    def char_encode(self, tokenList):
        ''' Returns id-encoded list from char-tokens using charIdx '''
        return list(map(self.char_to_id, tokenList))

    def char_decode(self, idList):
        ''' Returns string of decoded char id list '''
        return ''.join(map(self.id_to_char, idList))

    # DATA ANALYSIS
    def _question_extractor(self, questions, textTokens):
        ''' Helper to find question in table building '''
        for q in questions:
            if q['is_impossible']:
                yield Question(q['id'], q['question'], None)
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
            answerTokens = self.word_tokenize(answer['text'])
            answerStart = answerTokens[0]
            answerLen = len(answerTokens)
            span = None
            # print(answerTokens)
            for loc, word in enumerate(textTokens):
                if (word==answerStart):
                    # print(textTokens[loc:(loc+answerLen)], end=' | ')
                    # print(textTokens[loc:(loc+answerLen)] == answerTokens)
                    # uses the first appearance of the token in text if len=1
                    if ((textTokens[loc:(loc+answerLen)] == answerTokens)
                        and answerLen > 1):
                        span = (loc, loc+answerLen)
                        break
                    else:
                        span = (loc, loc)
            # character-tokenize question text
            # qTokens = self.char_tokenize(q['question'] + self.endToken)
            qTokens = self.char_tokenize('the')
            qTokenIds = self.char_encode(qTokens)
            yield Question(q['id'], qTokenIds, span)

    def _document_extractor(self, category):
        ''' Helper to find documents in table building '''
        title = category['title']
        for docId, document in enumerate(category['paragraphs']):
            text = document['context']
            textTokens = self.word_tokenize(text)
            questions = document['qas']
            questionIdx = {i : qObj for i, qObj
                           in enumerate(self._question_extractor(questions,
                                                                 textTokens))}
            textIds = self.word_encode(textTokens)
            yield Document(docId, title, textIds, questionIdx)

    def build(self, squadPath):
        ''' Builds SearchTable from squad file under squadPath'''
        print(colored('BUILDING SEARCH TABLE', 'yellow'))
        print(colored('Analyzing files:', 'red'), end='\r')
        with open(squadPath, 'r') as squadFile:
            data = json.load(squadFile)['data']
            self.categoryIdx = {category['title'] :
                                list(self._document_extractor(category))
                                for category in tqdm(data, leave=False)}
        print(colored('Complete: Analyzing files', 'cyan'))
        print(colored('Indexing files:', 'red'), end='\r')
        # TODO: build inverted index
        print(colored('Complete: Indexing files', 'cyan'))
        self.initialized = True
        print(colored('SEARCH TABLE BUILT', 'green'))
        return True
