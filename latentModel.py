'''
The latent model attempts to approximate the CLS vector embedding of questions
given their span and the text. Uses decoder RNN over GPT-2 token embeddings
with final binary dimension classifying token as element in or out of span.
Uses ReLu dense final layer on cell state of RNN followed by norming by mean
of GPT-2 embeddings (to accomodate negative values) to output embedding vector
approxmation of question relating to span.
'''

from structs import SearchTable

x = SearchTable(loadPath='SearchTable')
