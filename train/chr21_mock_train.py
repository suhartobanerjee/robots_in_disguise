import sys
sys.path.append("../src/")
import train
import gzip
from transformers import AutoTokenizer, AutoModel
import torch
from torch.utils.data import Dataset, DataLoader
import logging
import umap
import matplotlib.pyplot as plt
import numpy as np
import os


# logging config
logging.basicConfig(level=logging.DEBUG, format = "%(asctime)s %(levelname)s: %(message)s")

# setting the CUDA_VISIBLE_DEVICES
os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['GPUS']
logging.debug(os.environ['CUDA_VISIBLE_DEVICES'])


model_name = 'gena-lm-bert-base'
tokenizer = AutoTokenizer.from_pretrained(f'AIRI-Institute/{model_name}')
logging.info("Tokenizer downloaded!")


def read_chr_data(chr_id: str) -> str:
    with gzip.open(f"../data/chr{chr_id}.txt.gz", 'rb') as target:
        data = str(target.read())
    logging.info(f"Chromosome file read: chr{chr_id}!")

    # picking up the conf calls only
    conf_str = "".join([x for x in data if x.isupper()])
    logging.info(f"Chromosome file processed: chr{chr_id}!")
    return conf_str


chr_to_read = ["19", "21"]
conf_str = "".join([read_chr_data(chr) for chr in chr_to_read])
#conf_str = "".join(list(map(read_chr_data, chr_to_read)))




# TRAINING AREA
# variables for training.
vocab_size = len(tokenizer)
embed_dim = 64
n_heads = 8
n_layers = 6
ff_dim = 512
max_seq_len = 512
dropout = 0.1
batch_size = 32
n_epochs = 5

train_model = train.Train(vocab_size=vocab_size,
                          embed_dim=embed_dim,
                          n_heads=n_heads,
                          n_layers=n_layers,
                          ff_dim=ff_dim,
                          max_seq_len=max_seq_len,
                          dropout=dropout,
                          n_epochs=n_epochs,
                          batch_size=batch_size)


# logging the hyperparameters
logging.info(f"The hyperparameters are : \n\n1. vocab_size : {vocab_size},\n2. embed_dim : {embed_dim},\n3. n_heads : {n_heads},\n4. n_layers : {n_layers},\n5. ff_dim : {ff_dim},\n6. max_seq_len : {max_seq_len},\n7. dropout : {dropout},\n8. batch_size : {batch_size},\n9. n_epochs : {n_epochs},\n")


# padding seq
def add_padding(seq_chunk):
    if len(seq_chunk) < max_seq_len:
        pad_len = max_seq_len - len(seq_chunk)
        seq_chunk.extend(pad_len * [3])

    return seq_chunk


#chunked_seq = [conf_str[i:i+max_seq_len] for i in range(0, len(conf_str), max_seq_len)]
#tokenise all the chunks

# tokenise the entire seq
tokens = tokenizer(conf_str)
tokens = tokens['input_ids']

# removing the 1 and 2 tokens from the seq
tokens.remove(1)
tokens.remove(2)
#tokens = tokens.pop(0)
#tokens = tokens.pop(len(tokens))

# chunk the tokens list [input_ids] into sublists
# with max len == max_seq_len
# doing till max-seq-len -2
chunked_tokens = [tokens[i:i + (max_seq_len - 2)] for i in range(0, len(tokens), (max_seq_len - 2))]

# inserting 1 and 2 tokens at the start / end
[chunk.insert(0, 1) for chunk in chunked_tokens]
[chunk.append(2) for chunk in chunked_tokens]



# tokenising the chunked seq into a 2d array
#tokens = [tokenizer(chunk) for chunk in chunked_seq]
#tokens = [chunk['input_ids'] for chunk in tokens]

# padding seq which have less than max_seq_len
padded_tokens = list(map(add_padding, chunked_tokens))
#padded_tokens = chunked_tokens

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
logging.info(f"Input data shape : {gen_seq.size()}")


enc_output = train_model.training_cycle(data = gen_seq)
logging.info(f"Encoder output shape {enc_output.size()}")


# taking the hidden dims
enc_output_cpu = enc_output.cpu()
hidden_dims = enc_output_cpu.detach().numpy()

# saving the encoder output to disk
f = gzip.GzipFile("../proc/chr19_21_embed_dims.npy.gz", "w")
np.save(file=f,
        arr=hidden_dims)
f.close()


#hidden_dims = hidden_dims.flatten()
logging.info("Retrieved hidden dims!")

# performing umap red
#umap_red = umap.UMAP()
#umap_emb = umap_red.fit_transform(hidden_dims)
#logging.info("UMAP fitted!")
#
#logging.info("Plotting UMAP and saving plot...")
## plotting umap
#fig = plt.figure()
#plt.scatter(umap_emb[:, 0], umap_emb[:, 1])
#plt.title("UMAP Projection")
#fig.savefig("../plots/umap.pdf")
