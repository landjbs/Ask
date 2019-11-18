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
    def __init__(self, loadPath=None):
        if self.loadPath:
            self.categoryIdx = self.load(loadPath)
            self.initialized = True
        else:
            self.categoryIdx = {}
            self.initialized = False

    def save(self, savePath):
        assert self.initialized, 'SearchTable must be initialized before save.'
        u.safe_make_folder(savePath)
        u.save_obj(self.categoryIdx, f'{savePath}/categoryIdx.sav')
        return True

    def load(self, loadPath):
        assert not self.initialized, 'Cannot load into initialized SearchTable.'
        assert u.path_exists(loadPath), f'Cannot find path "{loadPath}".'
        self.categoryIdx = u.load_obj(f'{loadPath}/categoryIdx.sav')
        self.initialized = True
        return True

    def _tokenize(text):
        ''' returns tokenized text. to improve '''
        return text.lower().split()

    def _question_generator(questions, textWords):
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

    def _document_generator(category):
        ''' Helper to find documents in table building '''
        title = category['title']
        for docId, document in enumerate(category['paragraphs']):
            text = document['context']
            textWords = tokenize(text)
            questions = document['qas']
            questionIdx = {i : qObj for i, qObj
                            in enumerate(question_generator(questions,
                                                            textWords))}
            yield Document(docId, title, text, questionIdx)

    def build(self, squadPath):
        ''' Builds SearchTable from squad file under squadPath'''
        with open(squadPath, 'r') as squadFile:
            data = json.load(squadFile)
            self.categoryIdx = {category['title'] :
                                list(_document_generator(category))
                                for category in data['data']}
        self.initialized = True
        return True

    def __str__(self):
        outStr = 'SearchTableObj\n'
        for category, contents in self.categoryIdx.items():
            outStr += f'\t{category} : {len(contents)}\n'
        return outStr
