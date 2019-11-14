class Question(object):
    ''' Single question about a doc stores '''
    

class Document(object):
    ''' Single document from any QA dataset '''
    def __init__(self, docId, category, text, questionIdx):
        self.docId = docId
        self.category = category
        self.text = text
        self.questionIdx = questionIdx


class SearchTable(object):
    ''' Wide column hashtable of Document objects for searching '''

    def __init__(self):
        self.topicIdx = {}
