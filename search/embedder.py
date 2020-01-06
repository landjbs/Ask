from bert_serving.client import BertClient


# bert-serving-start -model_dir /Users/landon/Desktop/bertLarge -num_worker=1

class Embedder(object):
    def __init__(self):
        self.vectorizer = BertClient()

    def vectorize(self, text):
        return self.vectorizer.encode([text])[0]
