class Question(object):
    '''
    Single question about a Document. Can be human or machine asked.
    Args:
        id:     String id of question. Same as that used to fetch it
                from doc.questions.
        text:   Id-encoded text of question.
        span:   Tuple of (start, end) span of question in document.
        asker:  Id of question asker.
    '''
