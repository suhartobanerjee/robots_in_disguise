library(data.table)
library(rhdf5)
library(stringr)
library(purrr)
library(Seurat)
library(ggplot2)
library(RColorBrewer)
library(clustree)


source("./methods.R")

# some initialisations
job_id <- 6531478
chr_names <- "all_chr"
tmp_plot <- "../plots/tmp_plot.pdf"


################################################################################
# reading in the matrices
result_list <- transpose_matrix(
    read_datasets(str_glue("../proc/{job_id}_datasets.h5"))
)
embed_dim <- result_list[[1]]
input_tokens <- result_list[[2]]
dim(embed_dim)
dim(input_tokens)
remove(result_list)


# reading in labels
labels_dt <- proc_labels(
    read_labels("../proc/{job_id}_token_labels.txt")
)
labels_dt


# extracting tokens
token_extractor <- extract_tokens(embed_dim)
result_list <- map(c(1:2), token_extractor)

cls_stack <- result_list[[1]]
chr_stack <- result_list[[2]]
dim(cls_stack)
#cls_stack[1:2, 1:5]
#class(cls_stack)
remove(result_list)


# creating counts matrix
# transposing the mat so the shape is
# shape = (dims, n_samples)
cls_stack_counts <- aperm(cls_stack, c(2,1))

cls_stack_dt <- as.data.table(cls_stack_counts)
cls_stack_dt[, idx := as.character(
                        str_glue("dim-{c(1:nrow(cls_stack_dt))}")
                        )
]
setcolorder(
    cls_stack_dt,
    "idx"
)
dims <- cls_stack_dt[, idx]
dims
#cls_stack_dt[, 1:5]
#table(is.na(cls_stack_dt))


cls_stack_df <- as.data.frame(cls_stack_dt)
#cls_stack_df[1:5,1:5]
rownames(cls_stack_df) <- cls_stack_df$idx
cls_stack_df$idx <- NULL
cls_stack_df[1:5,1:5]


cls_so <- CreateSeuratObject(counts = cls_stack_df)
cls_so
#cls_so@assays$RNA@counts[1:5, 1:5]


#cls_so <- NormalizeData(cls_so, normalization.method = "LogNormalize", scale.factor = 10000)


# Identify the 10 most highly variable genes
cls_so <- FindVariableFeatures(cls_so, selection.method = "vst", nfeatures = 64)
VariableFeatures(cls_so)
top10 <- head(VariableFeatures(cls_so), 10)
top10

# plot variable features with and without labels
plot1 <- VariableFeaturePlot(cls_so)
plot1 <- LabelPoints(plot = plot1, points = top10, repel = TRUE)
plot1 <- plot1 + ggtitle("Highly variable dimensions")
ggsave(str_glue("../plots/{job_id}_highly_variable_dimensions.pdf"))



cls_so <- ScaleData(cls_so,
                    do.scale = F,
                    do.center = F,
                    features = dims
)
cls_so@assays$RNA@scale.data[1:5,1:5]


################################################################################
# pca
cls_so <- RunPCA(cls_so, npcs = 64, features = dims)

pca_plot <- DimPlot(cls_so, reduction = "pca") + NoLegend()
ggsave(str_glue("../plots/{job_id}_pca.pdf"))

elbw_plot <- ElbowPlot(cls_so)
ggsave(str_glue("../plots/{job_id}_elbow_plot.pdf"))


cls_so <- FindNeighbors(cls_so, dims = 1:18)
cls_so <- FindClusters(cls_so, resolution = 0.5)
head(Idents(cls_so), 5)


################################################################################
# umap
cls_so <- RunUMAP(cls_so, dims = 1:18)
umap_plot <- DimPlot(cls_so, reduction = "umap", label = T)
ggsave(str_glue("../plots/{job_id}_umap_0.5.pdf"))


# save rds
saveRDS(cls_so, file = str_glue("../proc/{job_id}_till_umap.rds"))



################################################################################
# cluster analysis

# load RDS
cls_so <- readRDS(str_glue("../proc/{job_id}_till_umap.rds"))


clusters <- c(5, 24, 0, 13, 14, 15)

pick_cluster <- select_cluster(cls_so)

clst_list <- map(clusters, pick_cluster)
length(clst_list)

tokens_pooler <- pool_cluster_tokens(input_tokens)

pooled_tokens_list <- map(clst_list, tokens_pooler)
length(pooled_tokens_list)
# pooled_tokens_list[[1]][1:15]

# pool all tokens from cluster
# length(pooled_tokens) + 2986 * 2 == 1528832

# pooled_freq <- table(pooled_tokens)
# pooled_freq[1:6]

pooled_freq_list <- map(pooled_tokens_list, table)

freq_dt_list <- map(pooled_freq_list, as.data.table)

freq_dt_list <- imap(freq_dt_list, function(dt, idx) dt[, cluster := clusters[idx]])

freq_dt <- reduce(freq_dt_list, rbind)


# finding the sum of the tokens in a cluster
freq_dt[, sum_tokens := sum(.SD$N), by = cluster]
setnames(
    freq_dt,
    "V1",
    "token"
)
freq_dt

# take the top n of each cluster
top_n <- 10
cutoff_dt <- freq_dt[, .SD[order(-N)][1:top_n], by = cluster]
cutoff_dt[, prop_N := N / sum_tokens]
cutoff_dt[, prop_labels := round(prop_N, 4)]
cutoff_dt


freq_plot <- ggplot(
    data = cutoff_dt,
    aes(x = token,
        y = factor(cluster),
        size = prop_N,
        label = prop_labels,
        color = prop_N
    )
) +
    geom_point() +
    geom_text(vjust = -2, size = 3, color = "black") +
    scale_color_gradient(low = "lightblue", high = "darkblue") +
    scale_y_discrete(name = "Clusters",
                     limits = factor(clusters)) +
    xlab("Tokens") +
    ggtitle(str_glue("Token Usage among clusters: Top {top_n} tokens")) +
    theme_bw()


# ggsave(tmp_plot, width = 14, height = 7)

ggsave(
    filename = str_glue("../plots/{job_id}_5_24_0_13-15_raw_token_usage_top{top_n}.pdf"),
    plot = freq_plot,
    width = 14,
    height = 7
)
ggsave(
    filename = str_glue("../plots/{job_id}_5_24_0_13-15_prop_token_usage_top{top_n}.pdf"),
    plot = freq_plot,
    width = 14,
    height = 7
)



# finding the dist of these top_n tokens in tensors
top_tokens <- as.integer(unique(cutoff_dt$token))
top_tokens

################################################################################
get_tensors <- get_cluster_tensors(input_tokens)
cluster_tensors <- map(clst_list, get_tensors)
str(cluster_tensors)

get_tokens_freq <- get_top_tokens_frequency(top_tokens)
filtered_cluster <- map(cluster_tensors, get_tokens_freq)
str(filtered_cluster)


filtered_cluster <- imap(filtered_cluster, function(dt, idx) dt[, cluster := clusters[idx]])
str(filtered_cluster)


occurence_dt <- reduce(filtered_cluster, rbind)
occurence_dt


tensor_freq_plot <- ggplot(
    data = occurence_dt,
    aes(x = factor(token),
        y = freq,
        color = factor(cluster)
    )
) +
    geom_violin() +
    stat_summary(fun=mean,
                    colour="darkred",
                    geom="crossbar",
                    width = 0.5
                    ) +
    xlab("tokens") +
    facet_grid(~ factor(cluster))

ggsave(tmp_plot,
       width = 20,
       height = 9
)

ggsave(filename = str_glue("../plots/{job_id}_5_24_0_13-15_token_top{top_n}_dist_tensors.pdf"),
       width = 20,
       height = 9
)


