'''
To read datasets into structs
'''

import json
from structs import Question, Document, SearchTable

def tokenize(text):
    ''' returns tokenized text. to improve '''
    return text.lower().split()

def question_generator(questions, textWords):
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
        answerWords = tokenize(answerText)
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
#
def document_generator(category):
        title = category['title']
        for docId, document in enumerate(category['paragraphs']):
            text = document['context']
            textWords = tokenize(text)
            questions = document['qas']
            questionIdx = {i : qObj for i, qObj
                            in enumerate(question_generator(questions,
                                                            textWords))}
            yield Document(docId, title, text, questionIdx)
