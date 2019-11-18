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

    def build(self, squadPath):
        ''' Builds SearchTable from squad file under squadPath'''
        with open(squadPath, 'r') as squadFile:
            data = json.load(squadFile)
            categoryIdx = {category['title'] : list(document_generator(category))
                            for category in data['data']}
            self.categoryIdx = SearchTable(categoryIdx)
        self.initialized = True
        return True

    def __str__(self):
        outStr = 'SearchTableObj\n'
        for category, contents in self.categoryIdx.items():
            outStr += f'\t{category} : {len(contents)}\n'
        return outStr
