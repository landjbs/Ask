class SearchTable(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.keywordIdx = {}
        self.clusterIdx = {}
        self.idIdx = {}
        self.pageNum = 0

    def __str__(self):
        return f'<SearchTable | pageNum={self.pageNum}>'

    def add_document(self, document):
        ''' Adds single document to kewordIdx, clusterIdx, and idIdx '''
        self.add_keyword_doc(document)
        self.add_cluster_doc(document)
        self.add_id_doc(document)

    def add_keyword_doc(self, document):
        pass

    def add_cluster_doc(self, document):
        pass

    def add_id_doc(self, i, document):
        self.idIdx.update({i : document})

    # loading data
