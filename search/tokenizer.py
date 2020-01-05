from transformers import GPT2LMHeadModel, GPT2Tokenizer

class Tokenizer(object):
    def __init__(self):
        self.tokenzier = GPT2Tokenizer.from_pretrained('gpt2')

    def get_tokens(self, text):
        ''' Returns word-tokenized text using gptTokenizer '''
        return self.gptTokenizer._tokenize(text.lower().strip())

    def encode_tokens(self, tokens):
        ''' Returns id-encoded list from word-tokens using gptTokenizer '''
        return self.gptTokenizer.convert_tokens_to_ids(tokenList)
