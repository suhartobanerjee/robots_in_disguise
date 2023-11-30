import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import polars as pl
import re
import os
from transformers import AutoTokenizer
model_name = 'gena-lm-bert-base'
tokenizer = AutoTokenizer.from_pretrained(f'AIRI-Institute/{model_name}')


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


def extract_token(_embed_dim, token_pos):
    token_stack = _embed_dim[:, token_pos:token_pos+1, :]
    token_stack = np.squeeze(token_stack)

    return token_stack


def df_from_dim_red(dim_red, red_method, labels, labels_col_name):
    umap_df = pl.from_numpy(dim_red, schema=[f"{red_method} 1", f"{red_method} 2"])
    umap_df = umap_df.with_columns(
        pl.Series(labels_col_name, labels),
        # pl.Series("idx", labels_idx).map_elements(
        #     lambda x: int(x.split("_")[1])
        # )
    )

    return umap_df


def plot_dim_red(dim_red_df, color_col, color_pal, plt_title, save_filename):
    fig, ax = plt.subplots(1, figsize=(6, 6))

    ax = sns.scatterplot(data=dim_red_df,
                         x=dim_red_df.columns[0],
                         y=dim_red_df.columns[1],
                         s=2,
                         hue=color_col,
                         palette=color_pal,
                         legend="full")

    plt.legend(markerscale=3)
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1.02, 1.02))
    plt.title(plt_title)
    fig.savefig(fname=save_filename, bbox_inches='tight')


def decode_sequences(tokens_list):
    sequence = list(map(tokenizer.decode, tokens_list))
    sequence = [re.sub(r"\[CLS\]|\[SEP\]", "", x) for x in sequence]
    sequence = [re.sub(r"$", r"\n\n", x) for x in sequence]

    return sequence


def clean_seq_fasta(filename):
    if os.path.exists(filename):
        os.remove(filename)


def write_seq_fasta(seq, label, x, filename="./test.fa"):
    if x == 0:
        clean_seq_fasta(filename)

    with open(filename, "a") as file:
        file.write(f"> {label}\n")
        file.write(seq)


def get_clusters(cluster_df, input_tokens, idx):
    cluster_filter_df = cluster_df.filter(
            pl.col("kmeans_clusters") == idx
            )
    cluster_filter_idx = cluster_filter_df['idx']

    sequence = decode_sequences(input_tokens[cluster_filter_idx])

    [write_seq_fasta(sequence[x],
                     cluster_filter_df['fasta_header'][x],
                     x,
                     filename=f"../proc/cluster_fasta/cluster_{idx}.fa") for x in range(len(sequence))]


