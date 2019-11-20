import torch
from os import listdir
from tqdm import tqdm
from transformers import GPT2Tokenizer, GPT2DoubleHeadsModel

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2DoubleHeadsModel.from_pretrained('gpt2')

def data_generator(folderPath='data/aclImdb/test'):
    negFolder = f'{folderPath}/neg'
    posFolder = f'{folderPath}/pos'
    for posFile, negFile in zip(listdir(posFolder), listdir(negFolder)):
        


# Add a [CLS] to the vocabulary (we should train it also!)
tokenizer.add_special_tokens({'cls_token': '[CLS]'})
model.resize_token_embeddings(len(tokenizer))  # Update the model embeddings with the new vocabulary size
print(tokenizer.cls_token_id, len(tokenizer))  # The newly token the last token of the vocabulary

encoded_choices = [tokenizer.encode(s) for s in choices]
cls_token_location = [tokens.index(tokenizer.cls_token_id) for tokens in encoded_choices]
input_ids = torch.tensor(encoded_choices).unsqueeze(0)  # Batch size: 1, number of choices: 2
mc_token_ids = torch.tensor([cls_token_location])  # Batch size: 1
outputs = model(input_ids, mc_token_ids=mc_token_ids)
print(outputs)
lm_prediction_scores, mc_prediction_scores = outputs[:2]
