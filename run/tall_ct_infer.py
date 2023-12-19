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
import multiprocessing as mp
import random
import polars as pl


# logging config
logging.basicConfig(level=logging.DEBUG, format = "%(asctime)s %(levelname)s: %(message)s")

# setting the CUDA_VISIBLE_DEVICES
os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['GPUS']
os.environ['TOKENIZERS_PARALLELISM'] = 'true'
logging.debug(os.environ['CUDA_VISIBLE_DEVICES'])
logging.info(f"Number of CUDA devices : {torch.cuda.device_count()}")


job_id = os.environ['JOB_ID']
logging.info(job_id)

# mp n_cores
n_cores = mp.cpu_count()
n_cores

###############################################################################
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
n_cores = mp.cpu_count()
n_cores

model_name = 'gena-lm-bert-base-t2t'
tokenizer = AutoTokenizer.from_pretrained(f'AIRI-Institute/{model_name}')
model = AutoModel.from_pretrained(f'AIRI-Institute/{model_name}',
                                  trust_remote_code=True)
model

# reading in the tsv file
tall_ct6 = pl.read_csv("../proc/tall_ct6_mutated_seq_amp_del_inv.tsv.gz",
                       has_header=True,
                       separator="\t"
                       )

# getting the seq column as a list
seq_arr = tall_ct6['V1']
len(seq_arr)
len(seq_arr[0])
seq_arr[0:2][:5]

# making a new dim for cells
seq_arr = [x for x in seq_arr]
len(seq_arr)
len(seq_arr[0][0:5])
seq_arr[2][0:5]

tokens = tokenizer(seq_arr[0])
tokens = tokens["input_ids"]
tokens["input_ids"][-1]


with mp.Pool(processes=n_cores-1) as pool:
    tokens_list = pool.map(tokenizer, seq_arr)

tokens_list = [x["input_ids"] for x in tokens_list]
len(tokens_list[0])

for chr in tokens_list:
    chr.remove(1)
    chr.remove(2)

def chunk_to_len(chr):
    return [chr[i:i + (max_seq_len - 2)] for i in range(0, len(chr), (max_seq_len - 2))]

max_seq_len = 512
chunked_tokens = [chunk_to_len(tokens) for tokens in tokens_list]
len((chunked_tokens))
len(chunked_tokens[1][0])
chunked_tokens[0][0][-1]


# inserting 1 and 2 tokens at the start / end
cells = tall_ct6['cell_name']
cell_list = []
def add_special_tokens(chr, cell):
    for chunk in chr:
        chunk.insert(0, 1)
        chunk.append(2)
        cell_list.append(cell)

    return chr


pre_proc_tokens = [add_special_tokens(chunked_tokens[i], cells[i]) for i in range(0, len(chunked_tokens))]
len(pre_proc_tokens[0][-1])
len(cell_list)
cell_list[0:5]

def add_padding(seq_chunk):
    if len(seq_chunk) < max_seq_len:
        pad_len = max_seq_len - len(seq_chunk)
        seq_chunk.extend(pad_len * [3])

    return seq_chunk


# pooling all the tensors together
proc_tokens = []
for chr in pre_proc_tokens:
    proc_tokens += chr

len(proc_tokens)
proc_tokens[-1][0]

padded_tokens = [add_padding(tokens) for tokens in proc_tokens]

len(padded_tokens)
len(padded_tokens[0])

tokens_tensor = torch.tensor(padded_tokens)
tokens_tensor.shape
tokens_tensor

# saving files
import h5py
with h5py.File(f"../proc/tall_ct6_proc_files.h5", "w") as f:
    f['tokens_tensor'] = tokens_tensor
    f['cell_list'] = cell_list

# run this with more mem
# also with batching

with torch.no_grad():
  output = model(tokens_tensor, output_hidden_states=True)
print(output.keys())
output['hidden_states'][-1].shape
output['hidden_states'][-1][0][0]

##############################################################################

def sample_nbp(seq: list, nbp: int) -> list:

    # getting the edge_idx
    # the logic is any random number between start and 
    # edge_idx will be the start of the subset
    # so that it always gives a valid subset
    edge_idx = len(seq) - nbp
    subset_start = random.sample(range(0, edge_idx), 1)[0]
    subset_end = subset_start + nbp

    return seq[subset_start:subset_end]


def read_chr_data(chr_id: str) -> str:
    if chr_id == "23":
        chr_id = "X"
    if chr_id == "24":
        chr_id = "Y"
    with gzip.open(f"../data/chr{chr_id}.txt.gz", 'rb') as target:
        data = str(target.read())
    logging.info(f"Chromosome file read: chr{chr_id}!")

    # picking up the conf calls only
    conf_str = [x for x in data if x.isupper()]
    conf_str = "".join(sample_nbp(conf_str, 10_000_000))
    logging.info(f"Chromosome file processed: chr{chr_id}!")
    logging.info(f"Length of sequence = {len(conf_str)}")
    return conf_str


chr_to_read = [str(x) for x in range(1, 25)]
#chr_to_read.append("X")
#chr_to_read.append("Y")

# parallelize
with mp.Pool(processes=n_cores) as pool:
    conf_list = pool.map(read_chr_data, chr_to_read)
#conf_list = [read_chr_data(chr) for chr in chr_to_read]
#conf_str = "".join(list(map(read_chr_data, chr_to_read)))




# TRAINING AREA
# variables for training.
vocab_size = len(tokenizer) + 25
embed_dim = 64
n_heads = 8
n_layers = 6
ff_dim = 512
max_seq_len = 512
dropout = 0.1
batch_size = 16
n_epochs = 10

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
#tokens_list = [tokenizer(chr)["input_ids"] for chr in conf_list]

# parallelize
with mp.Pool(processes=n_cores) as pool:
    tokens_list = pool.map(tokenizer, conf_list)

tokens_list = [x["input_ids"] for x in tokens_list]
logging.info("Sequence tokenized!")


# removing the 1 and 2 tokens from the seq
for chr in tokens_list:
    chr.remove(1)
    chr.remove(2)
#[tokens.remove(1) for tokens in tokens_list]
#[tokens.remove(2) for tokens in tokens_list]
#tokens = tokens.pop(0)
#tokens = tokens.pop(len(tokens))

# chunk the tokens list [input_ids] into sublists
# with max len == max_seq_len
# doing till max-seq-len -2
def chunk_to_len(chr):
    return [chr[i:i + (max_seq_len - 3)] for i in range(0, len(chr), (max_seq_len - 3))]

chunked_tokens = [chunk_to_len(chr) for chr in tokens_list]

labels = []
# inserting 1 and 2 tokens at the start / end
def add_special_tokens(chr, chr_id):
    for chunk in chr:
        chunk.insert(0, 1)
        chunk.insert(1, 32_000 + int(chr_id))
        chunk.append(2)
        labels.append("chr" + chr_id)

    return chr


pre_proc_tokens = [add_special_tokens(chunked_tokens[i], chr_to_read[i]) for i in range(0, len(chunked_tokens))]

# writing the labels to file
with open(f"../proc/{job_id}_token_labels.txt", "w") as file:
    file.write(str(labels))



proc_tokens = []
for chr in pre_proc_tokens:
    proc_tokens += chr



# tokenising the chunked seq into a 2d array
#tokens = [tokenizer(chunk) for chunk in chunked_seq]
#tokens = [chunk['input_ids'] for chunk in tokens]

# padding seq which have less than max_seq_len
padded_tokens = [add_padding(tokens) for tokens in proc_tokens]
#padded_tokens = chunked_tokens

#saving the padded_tokens to file to later work on it
f = gzip.GzipFile(f"../proc/{job_id}_input_tokens_matrix.npy.gz", "w")
np.save(file=f,
        arr=padded_tokens)
f.close()


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
f = gzip.GzipFile(f"../proc/{job_id}_all_chr_embed_dims.npy.gz", "w")
np.save(file=f,
        arr=hidden_dims)
f.close()


#hidden_dims = hidden_dims.flatten()
logging.info("Retrieved hidden dims! Program complete!!!")

