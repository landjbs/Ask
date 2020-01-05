from tqdm import tqdm

from search.document import Document
from search.question import Question

class SearchTable(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.keywordIdx = {}
        self.clusterIdx = {}
        self.idIdx = {}
        self.pageNum = 0

        # processing modules
        self.tokenizer = None
        self.keyworder = None
        self.embedder = None

    def __str__(self):
        return f'<SearchTable | pageNum={self.pageNum}>'

    # DATA LOADING
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

    def load_squad_file(self, path):
        ''' Loads squad file from path into various idxs '''
        with open(path, 'r') as squadFile:
            data = json.load(squadFile)['data']
            for category in tqdm(data, leave=False):
                title = category['title']
                


    # DATA YIELDING
    def gen_qa_batch(self, n, maxLen):
        '''
        Genereates batch tensor of size n with form (n, maxLen, )
        '''
        pass
