from collections import Counter
from flashtext import KeywordProcessor

class Keyworder(object):
    '''
    Keyworder gathers language data and maps raw document into dict of
    tokens scores. It gathers token distributions statistics from text file.
    '''
    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer
