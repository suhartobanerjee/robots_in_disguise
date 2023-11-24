import numpy as np
import umap
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import gzip
import polars as pl
import sklearn.cluster as cluster
import hdbscan
from transformers import AutoTokenizer
import re
import os
import multiprocessing as mp

model_name = 'gena-lm-bert-base'
tokenizer = AutoTokenizer.from_pretrained(f'AIRI-Institute/{model_name}')
print("All libraries imported!")

## 1. Encoder outputs

job_id = "6531478"
chr_names = "all_chr"

f = gzip.GzipFile(f"../proc/{job_id}_{chr_names}_embed_dims.npy.gz", "r")
embed_dim = np.load(f)
f.close()
embed_dim.shape


## 2. The labels

with open(f"../proc/{job_id}_token_labels.txt") as file:
    labels = file.read()
    print("labels read!")

def proc_labels(labels):
    labels = labels.replace("[", "")
    labels = labels.replace("]", "")
    labels = labels.replace("\'", "")
    labels = list(labels.split(","))

    # removing whitespace before string
    labels = [chr.strip() for chr in labels]
    old_labels = labels

    # renaming chrX and chrY
    labels = [chr.replace("chr23", "chrX") for chr in labels]
    labels = [chr.replace("chr24", "chrY") for chr in labels]

    labels_idx = [None] * len(labels)
    for idx, label in enumerate(labels):
        labels_idx[idx] = labels[idx] + "_" + str(idx)

    return (labels, labels_idx, old_labels)


labels, labels_idx, old_labels = proc_labels(labels)


## 3. Input tokens

f = gzip.GzipFile(f"../proc/{job_id}_input_tokens_matrix.npy.gz", "r")
input_tokens = np.load(f)
input_tokens.shape


## Token Extraction



def token_extractor(token_pos):

    def extract_token(embed_dim: []):
        token_stack = np.zeros((embed_dim.shape[0], embed_dim.shape[2]))
        for x in range(embed_dim.shape[0]):
            curr_stack = embed_dim[x:x+1,
                                   token_pos:token_pos+1,
                                   :].flatten()
            token_stack = np.vstack((token_stack, curr_stack))

        # removing the 0 rows
        token_stack = token_stack[~np.all(token_stack == 0, axis=1)]

        return token_stack

    return extract_token


cls_extractor = [token_extractor(x) for x in (0, 1)]
cls_extractor


def smap(f, embed_dim=embed_dim):
    return f(embed_dim)


# mp n_cores
n_cores = mp.cpu_count()

# parallelize
with mp.Pool(processes=n_cores) as pool:
    result_cls = pool.map_async(token_extractor(0), embed_dim)
    result_chr = pool.map_async(token_extractor(1), embed_dim)


cls_stack = result_cls.get()
chr_stack = result_chr.get()



## decode sequences

def decode_sequences(tokens_list):
    sequence = list(map(tokenizer.decode, tokens_list[0:5]))
    sequence = [re.sub(r"\[CLS\]|\[SEP\]", "", x) for x in sequence]
    sequence = [re.sub(r"$", r"\n\n", x) for x in sequence]

    return sequence


sequence = decode_sequences(input_tokens[0:5])
print(len(sequence))
sequence[0:5]


def clean_seq_fasta(filename):
    if os.path.exists(filename):
        os.remove(filename)


def write_seq_fasta(seq, label, x, filename="./test.fa"):
    if x == 0:
        clean_seq_fasta(filename)

    with open(filename, "a") as file:
        file.write(f"> {label}\n")
        file.write(seq)


[write_seq_fasta(sequence[x], labels_idx[x], x) for x in range(len(sequence))]





