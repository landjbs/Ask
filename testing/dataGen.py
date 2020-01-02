from search.document import Document
from search.question import Question


# nounList = [chr(i) for i in range(65, 91)]
# adjList = [chr(i) for i in range(97, 123)]
#
#
# def gen_example(i, noun, adj):
#     ''' Gens single document with a question '''
#     c = f'{noun} is very {adj}.'
#     q = f'what is {noun}.'
#     ques = Question(0, q, (2, 3))
#     doc = Document(i, {}, {0: ques}, [], None, c)
#     return doc
