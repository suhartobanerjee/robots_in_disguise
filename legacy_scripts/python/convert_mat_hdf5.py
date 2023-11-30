import h5py
import numpy as np
import gzip
import explore_methods
from explore_methods import *

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

labels
labels, labels_idx, old_labels = proc_labels(labels)


## 3. Input tokens

f = gzip.GzipFile(f"../proc/{job_id}_input_tokens_matrix.npy.gz", "r")
input_tokens = np.load(f)
f.close()
input_tokens.shape


# create an hdf5 file
with h5py.File(f"../proc/6531478_datasets.h5", "w") as f:
    f['embed_dim'] = embed_dim
    f['input_tokens'] = input_tokens


# checking if file writing was succesful
file = h5py.File(f"../proc/6531478_datasets.h5", "r")
file['embed_dim'].shape 
file['embed_dim'][0,0,0]
file['input_tokens'].shape
file['input_tokens'][0,4]
file.close()
