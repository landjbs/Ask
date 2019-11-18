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

    def get_questions(self):
        ''' Gets list of questions pertaining to document '''
        return self.questionIdx.values()


class SearchTable(object):
    ''' Wide column hashtable of Document objects for searching '''
    def __init__(self):
        self.topicIdx = {}
