from bert_serving.client import BertClient

class Embedder(object):
    def __init__(self):
        self.vectorizer = BertClient()

    def vectorize(self, text):
        return self.vectorizer.encode([text])[0]
