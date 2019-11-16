class Question(object):
    ''' Single question about a doc stores '''
    def __init__(self, questionId, text, span, is_impossible):
        self.questionId = questionId
        self.text = text
        self.span = span
        self.is_impossible = is_impossible


class Document(object):
    ''' Single document from any QA dataset '''
    def __init__(self, docId, category, text, questionIdx):
        self.docId = docId
        self.category = category
        self.text = text
        self.questionIdx = questionIdx

    def get_questions(self):
        ''' Gets list of questions pertaining to document '''
        return self.questionIdx.values()


class SearchTable(object):
    ''' Wide column hashtable of Document objects for searching '''
    def __init__(self):
        self.topicIdx = {}
