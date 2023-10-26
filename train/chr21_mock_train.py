import sys
sys.path.append("../src/")
import model


import torch.optim as optim
import gzip
from transformers import AutoTokenizer, AutoModel
import torch




model_name = 'gena-lm-bert-base'
tokenizer = AutoTokenizer.from_pretrained(f'AIRI-Institute/{model_name}')


with gzip.open("../data/chr21.txt.gz", 'rb') as target:
    data = str(target.read())

print((data[1:10]))
print(type(data))


conf_pos = list(filter(lambda c: c.isupper(), data))
conf_str = ''.join(conf_pos)
print((conf_str[1:10]))


tokens = tokenizer(conf_str[1:100])
tokens_tensor = torch.tensor(tokens['input_ids']).unsqueeze(0)


print(tokens_tensor)




src_vocab_size = 270000
embed_dim = 4
num_heads = 2
num_layers = 1
ff_dim = 10
max_seq_length = 36
dropout = 0.1

gbert = model.GBERT(src_vocab_size, embed_dim, num_heads, num_layers, ff_dim, max_seq_length, dropout)

optimizer = optim.Adam(gbert.parameters(),)
gbert.train()


for epoch in range(2):
    optimizer.zero_grad()
    output = gbert(tokens_tensor)
    optimizer.step()
    print(f"Epoch: {epoch+1}, Output: \n{output}")
