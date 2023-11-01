import sys
sys.path.append("../src/")
import train
import gzip
from transformers import AutoTokenizer, AutoModel
import torch
from torch.utils.data import Dataset, DataLoader
import logging


# logging config
logging.basicConfig(level=logging.INFO)


model_name = 'gena-lm-bert-base'
tokenizer = AutoTokenizer.from_pretrained(f'AIRI-Institute/{model_name}')
logging.info("Tokenizer downloaded!")


with gzip.open("../data/chr21.txt.gz", 'rb') as target:
    data = str(target.read())
logging.info("Chromosome file read!")


# picking up the conf calls only
conf_pos = list(filter(lambda c: c.isupper(), data))
conf_str = ''.join(conf_pos[10000:10200])
logging.info("Chromosome file processed!")



# TRAINING AREA
# variables for training.
vocab_size = len(tokenizer)
embed_dim = 10
n_heads = 2
n_layers = 2
ff_dim = 256
max_seq_len = 50
dropout = 0.1
batch_size = 16
n_epochs = 5000

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
        seq_chunk.extend(pad_len * [3])

    return seq_chunk


chunked_seq = [conf_str[i:i+max_seq_len] for i in range(0, len(conf_str), max_seq_len)]

# tokenise the entire seq
tokens = tokenizer(conf_str)
# chunk the tokens list [input_ids] into sublists
# with max len == max_seq_len
chunked_tokens = [tokens['input_ids'][i:i + max_seq_len] for i in range(0, len(tokens['input_ids']), max_seq_len)]

# tokenising the chunked seq into a 2d array
#tokens = [tokenizer(chunk) for chunk in chunked_seq]
#tokens = [chunk['input_ids'] for chunk in tokens]

# padding seq which have less than max_seq_len
padded_tokens = list(map(add_padding, chunked_tokens))


# converting into tensor.
tokens_tensor = torch.tensor(padded_tokens)


class GenomeSequences(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def show(self):
        return self.data

    def size(self):
        return self.data.size()

gen_seq = GenomeSequences(tokens_tensor)
logging.info("Dataset processed! Beginning training ...")


enc_output = train_model.training_cycle(data = gen_seq)
logging.info(f"Encoder output shape {enc_output.size()}")
