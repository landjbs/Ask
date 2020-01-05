from transformers import GPT2Tokenizer

class Tokenizer(object):
    def __init__(self):
        self.tokenzier = GPT2Tokenizer.from_pretrained('gpt2')

    # cleaning
    def clean_text(self, s):
        ''' Returns cleaned text '''
        return s.strip().lower()

    # tokenization
    def get_tokens(self, text):
        ''' Returns word-tokenized text using gptTokenizer '''
        return self.tokenzier._tokenize(text.lower().strip())

    def encode_tokens(self, tokens):
        ''' Returns id-encoded list from word-tokens using gptTokenizer '''
        return self.tokenzier.convert_tokens_to_ids(tokens)

    def decode_ids(self, ids):
        ''' Returns list of str tokens from list of id encodings '''
        return self.tokenzier.convert_tokens_to_ids(ids)

    def string_to_ids(self, s):
        ''' Converts raw text string to list of ids '''
        return self.encode_tokens(self.get_tokens(self.clean_text(s)))

    def ids_to_string(self, ids):
        return ''.join(self.decode_ids(ids))
