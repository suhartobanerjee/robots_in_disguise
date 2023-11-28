import numpy as np
import umap
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import gzip
import polars as pl
import sklearn.cluster as cluster
import multiprocessing as mp
import time
import sklearn
import colorcet as cc
import importlib
import explore_methods
from explore_methods import *
importlib.reload(explore_methods)

print("All libraries imported!")
tmp_plot = "../plots/tmp_plot.pdf"


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

labels, labels_idx, old_labels = proc_labels(labels)


## 3. Input tokens

f = gzip.GzipFile(f"../proc/{job_id}_input_tokens_matrix.npy.gz", "r")
input_tokens = np.load(f)
f.close()
input_tokens.shape


## Token Extraction


# mp n_cores
n_cores = mp.cpu_count()
n_cores

# sequential
# use a map function here maybe?
result_list = [extract_token(embed_dim, x) for x in [0, 1]]

cls_stack = result_list[0]
chr_stack = result_list[1]
cls_stack.shape


## PCA check

pca_red = sklearn.decomposition.PCA(n_components=10)

# cls token
cls_pca = pca_red.fit_transform(cls_stack)
cls_pca.shape
var_pca = [round(x, 4) for x in pca_red.explained_variance_ratio_]
var_pca

cls_pca_df = df_from_dim_red(cls_pca[:, 0:2], "pca", labels, "chr")
cls_pca_df

cls_pca_df = cls_pca_df.rename(
        {
            "pca 1": f"pca 1: {var_pca[0]}",
            "pca 2": f"pca 2: {var_pca[1]}"
        }
        )
cls_pca_df

#plotting
plot_dim_red(dim_red_df=cls_pca_df,
             color_col="chr",
             color_pal=palette,
             plt_title="CLS Token: PCA",
             save_filename=f"../plots/{job_id}_{chr_names}_cls_pca.pdf"
             )


# chr token 
chr_pca = pca_red.fit_transform(chr_stack)
chr_pca.shape
var_pca = [round(x, 4) for x in pca_red.explained_variance_ratio_]
var_pca

chr_pca_df = df_from_dim_red(chr_pca[:, 0:2], "pca", labels, "chr")
chr_pca_df

chr_pca_df = chr_pca_df.rename(
        {
            "pca 1": f"pca 1: {var_pca[0]}",
            "pca 2": f"pca 2: {var_pca[1]}"
        }
        )
chr_pca_df

#plotting
plot_dim_red(dim_red_df=chr_pca_df,
             color_col="chr",
             color_pal=palette,
             plt_title="chr Token: PCA",
             save_filename=f"../plots/{job_id}_{chr_names}_chr_pca.pdf"
             )




## Umap reduction

umap_red = umap.UMAP()
umap_cls = umap_red.fit_transform(cls_pca)
umap_chr = umap_red.fit_transform(chr_pca)


# making a df to use with seaborn
umap_cls_df = df_from_dim_red(umap_cls, "umap", labels, "chr")
umap_chr_df = df_from_dim_red(umap_chr, labels, labels_idx)
umap_cls_df

plot_dim_red(dim_red_df=umap_cls_df,
             color_col="chr",
             plt_title="CLS Token: UMAP",
             save_filename=tmp_plot
             )

## agglomerative clustering

ag_clustering = cluster.AgglomerativeClustering()
ag_clusters = ag_clustering.fit_predict(cls_pca)
len(set(ag_clusters))
set(ag_clusters)

# making a df to use with seaborn
agclst_cls_df = df_from_dim_red(umap_cls, "umap", ag_clusters, "ag_clst")

# plotting
plot_dim_red(dim_red_df=agclst_cls_df,
             color_col="ag_clst",
             plt_title="CLS Token: AgglomerativeClustering",
             save_filename=tmp_plot
             )


## HDBSCAN clustering

hdbscan_red = sklearn.cluster.HDBSCAN(n_jobs=n_cores
                                      )

cls_clusters_hdbscan = hdbscan_red.fit_predict(cls_pca)
len(set(cls_clusters_hdbscan))
set(cls_clusters_hdbscan)
len(cls_clusters_hdbscan)
cls_clusters_hdbscan[0:5]
hdbscan_red.fit(cls_stack)



# making a df to use with seaborn
hdbs_cls_df = df_from_dim_red(umap_cls, "hdbscan", cls_clusters_hdbscan, "hdbscan_clusters")
hdbs_cls_df
hdbs_cls_df = hdbs_cls_df.filter(
    pl.col("hdbscan_clusters") != -1
    )
n_clusters = len(set(hdbs_cls_df["hdbscan_clusters"]))
hdbs_cls_df
#hdbs_cls_df.filter(
#        pl.col("hdbscan_clusters") == 17
#        )


# plotting
palette = sns.color_palette(cc.glasbey_light, n_colors=n_clusters)
#col_pal = sns.color_palette("Paired" , 24)
plot_dim_red(dim_red_df=hdbs_cls_df,
             color_col="hdbscan_clusters",
             color_pal=palette,
             plt_title="CLS Token: HDBSCAN Clustering",
             save_filename=tmp_plot
             #save_filename=f"../plots/{job_id}_{chr_names}_cls_hdbscan.pdf"
             )


## Kmeans
kmeans = cluster.KMeans(n_clusters=4)
kmeans.fit(cls_pca)

len(set(kmeans.labels_))
n_clusters = len(set(kmeans.labels_))

kmeans_cls_df = df_from_dim_red(umap_cls, "kmeans", kmeans.labels_, "kmeans_clusters")
kmeans_cls_df


palette = sns.color_palette(cc.glasbey_light, n_colors=n_clusters)
plot_dim_red(dim_red_df=kmeans_cls_df,
             color_col="kmeans_clusters",
             color_pal=palette,
             plt_title="CLS Token: Kmeans Clustering",
             #save_filename=tmp_plot
             save_filename=f"../plots/{job_id}_{chr_names}_cls_kmeans.pdf"
             )

# decode sequences

kmeans_cls_df = kmeans_cls_df.with_columns(
        pl.Series("idx", labels_idx).map_elements(
             lambda x: int(x.split("_")[1])
        )
        )
kmeans_cls_df
len(input_tokens[0])
kmeans_cls_df.write_csv(file="./kmeans_cls_df.tsv", separator="\t")

# adding a unique fasta_header
kmeans_cls_df = kmeans_cls_df.with_columns(
        pl.Series("fasta_header", labels_idx)
        )


# writing each cluster sequences to fasta file
[get_clusters(kmeans_cls_df, input_tokens, x) for x in set(kmeans_cls_df['kmeans_clusters'])]

