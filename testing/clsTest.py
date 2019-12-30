import torch
from os import listdir
from tqdm import tqdm
from transformers import GPT2Tokenizer, GPT2DoubleHeadsModel, TrainDataset


dataset = TrainDataset()
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2DoubleHeadsModel.from_pretrained('gpt2')
tokenizer.add_special_tokens({'cls_token': '[CLS]'})
model.resize_token_embeddings(len(tokenizer))

def data_generator(folderPath='data/aclImdb/test'):
    negFolder = f'{folderPath}/neg'
    posFolder = f'{folderPath}/pos'
    for posPath, negPath in tqdm(zip(listdir(posFolder), listdir(negFolder))):
        with open(f'{posFolder}/{posPath}', 'r') as posFile:
            posText = posFile.read()
        with open(f'{negFolder}/{negPath}', 'r') as negFile:
            negText = negFile.read()
    posText = '[CLS]' + posText
    negText = '[CLS]' + negText
    cls_token_location = [0, 0]
    input_ids = torch.tensor(encoded_choices).unsqueeze(0)

# Add a [CLS] to the vocabulary (we should train it also!)

  # Update the model embeddings with the new vocabulary size
print(tokenizer.cls_token_id, len(tokenizer))  # The newly token the last token of the vocabulary
# choices = ['hi [CLS]', 'yo what up [CLS]']
choices = ['[CLS]']
encoded_choices = [tokenizer.encode(s) for s in choices]
print(encoded_choices)
cls_token_location = [tokens.index(tokenizer.cls_token_id) for tokens in encoded_choices]
print(cls_token_location)
input_ids = torch.tensor(encoded_choices).unsqueeze(0)  # Batch size: 1, number of choices: 2
print(input_ids)
mc_token_ids = torch.tensor([cls_token_location])  # Batch size: 1
outputs = model(input_ids, mc_token_ids=mc_token_ids)
print(outputs)
lm_prediction_scores, mc_prediction_scores = outputs[:2]
