class Question(object):
    '''
    Single question about a Document. Can be human or machine asked.
    Args:
        id:     String id of question. Same as that used to fetch it
                from doc.questions.
        text:   Id-encoded text of question.
        span:   Tuple of (start, end) span of question in document.
        asker:  String id of question asker.
    '''
    def __init__(self, id, text, span, asker):
        self.id = id
        self.text = text
        self.span = span
        self.asker = asker

    def __str__(self):
        return f'<Question | {self.text}>'

    def answer(self, doc, model):
        ''' Answers unanswered question using model '''
        # TODO: imp
        pass
