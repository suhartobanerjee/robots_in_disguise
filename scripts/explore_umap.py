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
import time
import sklearn

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
f.close()
input_tokens.shape


## Token Extraction


def extract_token(_embed_dim, token_pos):
    token_stack = _embed_dim[:, token_pos:token_pos+1, :]
    token_stack = np.squeeze(token_stack)

    return token_stack


# mp n_cores
n_cores = mp.cpu_count()

# sequential
# use a map function here maybe?
tstart = time.time()
cls_stack = extract_token(embed_dim, 0)
print(time.time() - tstart)



len(result_list)
cls_stack = result_list[0]
chr_stack = result_list[1]
chr_stack.shape


## Umap reduction

umap_red = umap.UMAP()
umap_embed_cls = umap_red.fit_transform(cls_stack)
umap_embed_chr = umap_red.fit_transform(chr_stack)


# making a df to use with seaborn
def df_from_umap(umap, labels, labels_col_name):
    umap_df = pl.from_numpy(umap, schema=["UMAP 1", "UMAP 2"])
    umap_df = umap_df.with_columns(
        pl.Series(labels_col_name, labels),
        # pl.Series("idx", labels_idx).map_elements(
        #     lambda x: int(x.split("_")[1])
        # )
    )

    return umap_df


umap_df_cls = df_from_umap(umap_embed_cls, labels, labels_idx)
umap_df_chr = df_from_umap(umap_embed_chr, labels, labels_idx)
umap_df_cls


# clustering the UMAP with hdbscan

hdbscan_red = sklearn.cluster.HDBSCAN(n_jobs=n_cores,
                                      min_cluster_size=30,
                                      min_samples=5
                                      )

cls_clusters_hdbscan = hdbscan_red.fit_predict(umap_embed_cls)
len(set(cls_clusters_hdbscan))
set(cls_clusters_hdbscan)
len(cls_clusters_hdbscan)
cls_clusters_hdbscan[0:5]
hdbscan_red.fit(cls_stack)



# making a df to use with seaborn
hdbs_cls_df = df_from_umap(umap_embed_cls, cls_clusters_hdbscan, "hdbscan_clusters")
hdbs_cls_df = hdbs_cls_df.filter(
    pl.col("hdbscan_clusters") != -1
    )


# plotting
def plot_umap(umap_df, color_col, plt_title, save_filename):
    fig, ax = plt.subplots(1, figsize=(6, 6))

    ax = sns.scatterplot(data=umap_df,
                         x="UMAP 1",
                         y="UMAP 2",
                         s=2,
                         hue=color_col)

    plt.legend(markerscale=3)
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1.02, 1.02))
    plt.title(plt_title)
    fig.savefig(fname=save_filename, bbox_inches='tight')


plot_umap(umap_df=hdbs_cls_df,
          color_col="hdbscan_clusters",
          plt_title="CLS Token: HDBSCAN Clustering",
          save_filename=f"../plots/{job_id}_{chr_names}_cls_hdbscan.pdf"
          )

# decode sequences

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





