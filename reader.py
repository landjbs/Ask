'''
To read datasets into structs
'''

import json
from structs import Question, Document, SearchTable

def tokenize(text):
    ''' returns tokenized text. to improve '''
    return text.split()


with open('data/inData/dev-v2.0.json', 'r') as squadFile:
    data = json.load(squadFile)
    for category in data['data']:
        title = category['title']
        for document in category['paragraphs']:
            text = document['context']
            textWord = text.split()
            questions = document['qas']
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
                answerWords = answerText.split()


                #
                # qObj = Question(q['id'], q['question'])
                # if question['is_impossible']:
                #         span = None
                # else:
                #     # supports answers and plausible_answers interchangably
                #     try:
                #         answerList = question['answers']
                #     except KeyError:
                #         answerList = question['plausible_answers']
                #     assert (len(answerList) == 1), f'{len(answerList)}'
                #     answerText = answerList[0]['text']
                #     answerTokens = tokenizer.tokenize(answerText)
                #     answerIds = tokenizer.convert_tokens_to_ids(answerTokens)
                #     #find span of answerText in paragraph
                #     spanLen = len(answerIds)
                #     found = False
                #     for idLoc, firstId in enumerate(paragraphIds):
                #         if (firstId == answerIds[0]):
                #             endLoc = idLoc + spanLen
                #             if (paragraphIds[idLoc : endLoc] == answerIds) and (not found):
                #                 startVec = np.zeros(shape=maxContextLen)
                #                 endVec = np.zeros(shape=maxContextLen)
                #                 startVec[idLoc] = 1
                #                 endVec[endLoc] = 1
                #                 startTargets.append(startVec)
                #                 endTargets.append(endVec)
                #                 found = True
                #                 break
                #     if not found:
                #         startTargets.append([0 for _ in range(maxContextLen)])
                #         endTargets.append([0 for _ in range(maxContextLen)])
