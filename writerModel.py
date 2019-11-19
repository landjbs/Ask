'''
Decodes latent vector into raw text. Trained before application to question
generation.
'''

from transformers import GPT2LMHeadModel, GPT2Tokenizer

# init models and tokenizer
model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model.eval()

while True:
    text = input('text: ')
    context_tokens = tokenizer.encode(text, add_special_tokens=False)
    print(context_tokens)

# def generate(text):
