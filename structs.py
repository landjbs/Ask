import re
import json
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
        self.char_to_id = lambda c : charIdx[c]
        self.charMatcher = re.compile("|".join(charList))
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
        return self.gptTokenizer._tokenize(text.lower().strip(),
                                           add_special_tokens=False)

    def word_encode(self, tokenList):
        ''' Returns id-encoded list from word-tokens using gptTokenizer '''
        return self.gptTokenizer.convert_tokens_to_ids(tokenList)

    def char_tokenize(self, text):
        ''' Returns char-tokenized text using charIdx '''
        return list(map(self.char_to_id,
                        self.charMatcher.findall(text.lower())))

    def embed(self, wordIds):
        ''' Embeds list of wordIds tokenized by gptTokenizer with gpt2Model '''
        # WARNING: DOES NOT YET WORK
        wordIds = torch.tensor(wordIds, dtype=torch.long, device=device)
        wordIds = wordIds.unsqueeze(0).repeat(1, 1)
        with torch.no_grad():
            outputs = self.gptModel(**inputs)
        return outputs

    # DATA ANALYSIS
    def _question_extractor(self, questions, textIds):
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
            answerIds = self.word_tokenize(answer['text'].lower().strip())
            answerStart = answerIds[0]
            answerLen = len(answerIds)
            span = None
            # print(f'{"-"*80}\nANSWER: {answerIds}')

            if answerStart in  textIds:
                print(True)
            else:
                print(False)

            # for loc, wordId in enumerate(textIds):
            #     if (wordId==answerStart):
            #         print(f'\t{textIds[loc:(loc+answerLen)]} | {textIds[loc:(loc+answerLen)] == answerIds}')
            #         if (textIds[loc:(loc+answerLen)] == answerIds):
            #             span = (loc, loc+answerLen)
            #             break
            # print(f'\n{"-"*80}')
            # character-tokenize question text
            qTokenIds = self.char_tokenize(q['question'] + self.endToken)
            yield Question(q['id'], qTokenIds, span)

    def _document_extractor(self, category):
        ''' Helper to find documents in table building '''
        title = category['title']
        for docId, document in enumerate(category['paragraphs']):
            text = document['context']
            textTokens = self.gptTokenizer._tokenize(text.lower().strip())
            questions = document['qas']
            questionIdx = {i : qObj for i, qObj
                           in enumerate(self._question_extractor(questions,
                                                                 textIds))}
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
