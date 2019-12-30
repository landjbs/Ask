class Document(object):
    '''
    A document is a page of text from any source in SearchTable.
    Args:
        id:         String of url or SQuAD id.
        tokens:     Dict of tokens on or relevant to page and
                    corresponding scores.
        questions:  Dict of questions asked about document.
        vec:        Cluster vector embedding of page text.
        path:       Path to text if it is stored in local memory.
    '''
    def __init__(self, id, tokens, questions, vec, path=None):
        self.id = id
        self.tokens = tokens
        self.questions = questions
        self.vec = vec
        self.path = path
        self.text = self.fetch_text(path)

    def __str__(self):
        return f'<DocumentObj | id={self.docId}>'

    def iter_questions(self):
        ''' Iterates over questions in doc, yielding tuple (text, span) '''
        for question in self.questions.values():
            yield (question.text, question.span)
