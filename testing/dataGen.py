from search.document import Document
from search.question import Question


nounList = [chr(i) for i in range(65, 91)]
adjList = [chr(i) for i in range(97, 123)]


def gen_example():
    ''' Gens single document with a question '''
    
