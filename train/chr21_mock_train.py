import sys
sys.path.append("../src/")
import train
import gzip
from transformers import AutoTokenizer, AutoModel
import torch




model_name = 'gena-lm-bert-base'
tokenizer = AutoTokenizer.from_pretrained(f'AIRI-Institute/{model_name}')


with gzip.open("../data/chr21.txt.gz", 'rb') as target:
    data = str(target.read())


# picking up the conf calls only
conf_pos = list(filter(lambda c: c.isupper(), data))
conf_str = ''.join(conf_pos)


tokens = tokenizer(conf_str[1:50000])
tokens_tensor = torch.tensor(tokens['input_ids']).unsqueeze(0)


# TRAINING AREA
# variables for training.
vocab_size = len(tokenizer)
embed_dim = 4
n_heads = 2
n_layers = 1
ff_dim = 10
max_seq_length = tokens_tensor.size(dim = 1)
dropout = 0.1
batch_size = 5
n_epochs = 150

train_model = train.Train(vocab_size = vocab_size,
                          embed_dim = embed_dim,
                          n_heads = n_heads,
                          n_layers = n_layers,
                          ff_dim = ff_dim,
                          max_seq_len = max_seq_length,
                          dropout = dropout,
                          n_epochs = n_epochs,
                          batch_size = batch_size)

train_model.training_cycle(data = tokens_tensor)
