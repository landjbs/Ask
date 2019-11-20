'''
Decodes latent vector into raw text. Trained before application to question
generation.
'''

import torch
from tqdm import trange
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from scipy.spatial.distance import euclidean
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# init models and tokenizer
model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.eval()

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (batch size x vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(dim=1, index=sorted_indices, src=sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits

def generate(model, length, context, num_samples=1, temperature=1, top_k=0,
             top_p=0.0, repetition_penalty=1, device='cpu'):
    context = torch.tensor(context, dtype=torch.long, device=device)
    context = context.unsqueeze(0).repeat(num_samples, 1)
    generated = context
    with torch.no_grad():
        for _ in trange(length, leave=False):
            inputs = {'input_ids': generated}
            outputs = model(**inputs)
            next_token_logits = outputs[0][:, -1, :] / (temperature if temperature > 0 else 1.)
            for i in range(num_samples):
                for _ in set(generated[i].tolist()):
                    next_token_logits[i, _] /= repetition_penalty
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
            if temperature == 0: # greedy sampling:
                next_token = torch.argmax(filtered_logits, dim=-1).unsqueeze(-1)
            else:
                next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
            generated = torch.cat((generated, next_token), dim=1)
    return generated

while True:
    text = input('text: ')
    context_tokens = tokenizer.encode(text, add_special_tokens=False)
    out = (generate(model, 20, context_tokens))
    out = out[:, len(context_tokens):].tolist()
    outText = text.strip() + ' '.join(tokenizer.decode(o, clean_up_tokenization_spaces=True) for o in out)
    print(f'{outText}\n{"-"*80}')
