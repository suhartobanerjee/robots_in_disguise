import sys
sys.path.append("../src/")
import model
import gzip
from transformers import AutoTokenizer, AutoModel
import torch


# reading in chr21
# taking the confident calls
with gzip.open("../data/chr21.txt.gz", 'rb') as target:
    data = str(target.read())
conf_pos = list(filter(lambda c: c.isupper(), data))
conf_str = ''.join(conf_pos)


model_name = 'gena-lm-bert-base'
tokenizer = AutoTokenizer.from_pretrained(f'AIRI-Institute/{model_name}')

tokens = tokenizer(conf_str[1:100])
tokens_tensor = torch.tensor(tokens['input_ids']).unsqueeze(0)

print(tokens_tensor)
print("\n")


mask = model.CreateMask()
masked_seq, original_labels, masked_indices = mask.add_mask_token(tokens_tensor)


print(masked_seq)
print("\n")
print(original_labels)
print(masked_indices)
