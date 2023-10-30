import sys
sys.path.append("../src/")
import train
import gzip
from transformers import AutoTokenizer, AutoModel
import torch
from torch.utils.data import DataLoader




model_name = 'gena-lm-bert-base'
tokenizer = AutoTokenizer.from_pretrained(f'AIRI-Institute/{model_name}')


with gzip.open("../data/chr21.txt.gz", 'rb') as target:
    data = str(target.read())


# picking up the conf calls only
conf_pos = list(filter(lambda c: c.isupper(), data))
conf_str = ''.join(conf_pos[0:100])




# TRAINING AREA
# variables for training.
vocab_size = len(tokenizer)
embed_dim = 256
n_heads = 8
n_layers = 12
ff_dim = 256
max_seq_len = 20
dropout = 0.1
batch_size = 16
n_epochs = 100

train_model = train.Train(vocab_size = vocab_size,
                          embed_dim = embed_dim,
                          n_heads = n_heads,
                          n_layers = n_layers,
                          ff_dim = ff_dim,
                          max_seq_len = max_seq_len,
                          dropout = dropout,
                          n_epochs = n_epochs,
                          batch_size = batch_size)


# padding seq
def add_padding(seq_chunk):
    if len(seq_chunk) < max_seq_len:
        pad_len = max_seq_len - len(seq_chunk)
        seq_chunk.extend(pad_len * [104])

    return seq_chunk


chunked_seq = [conf_str[i:i+max_seq_len] for i in range(0, len(conf_str), max_seq_len)]


tokens = [tokenizer(chunk) for chunk in chunked_seq]
tokens = [chunk['input_ids'] for chunk in tokens]

# padding seq which have less than max_seq_len
tokens = list(map(add_padding, tokens))

tokens_tensor = torch.tensor(tokens)
print(tokens_tensor.size())

data_loader = []
for seq in tokens_tensor:
    data_loader.extend(DataLoader(seq, batch_size = batch_size, shuffle = True))
print(len(data_loader))
print(data_loader)
#train_model.training_cycle(data = tokens_tensor)
