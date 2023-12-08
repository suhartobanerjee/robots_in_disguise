library(data.table)
library(rhdf5)
library(stringr)
library(purrr)
library(Seurat)
library(ggplot2)
library(RColorBrewer)
library(clustree)
library(rjson)
library(furrr)
library(hash)


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

# read in the vocab
vocab_dt <- fread("../proc/bpe_vocab.tsv")
setkey(vocab_dt, token)
vocab_dt

################################################################################
# extracting tokens
token_extractor <- extract_tokens(embed_dim)
result_list <- map(c(1:2), token_extractor)

cls_stack <- result_list[[1]]
chr_stack <- result_list[[2]]
dim(cls_stack)
#cls_stack[1:2, 1:5]
#class(cls_stack)
remove(result_list)

################################################################################


################################################################################
# creating counts matrix
# transposing the mat so the shape is
# shape = (dims, n_samples)
# cls stack here
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


########################################
# chr stack here
chr_stack_counts <- aperm(chr_stack, c(2,1))

chr_stack_dt <- as.data.table(chr_stack_counts)
chr_stack_dt[, idx := as.character(
                        str_glue("dim-{c(1:nrow(chr_stack_dt))}")
                        )
]
setcolorder(
    chr_stack_dt,
    "idx"
)
dims <- chr_stack_dt[, idx]
dims
#chr_stack_dt[, 1:5]
#table(is.na(chr_stack_dt))


chr_stack_df <- as.data.frame(chr_stack_dt)
#chr_stack_df[1:5,1:5]
rownames(chr_stack_df) <- chr_stack_df$idx
chr_stack_df$idx <- NULL
chr_stack_df[1:5,1:5]

################################################################################

################################################################################
# For seurat, go to seurat_run.R file
################################################################################

# load RDS
cls_so <- readRDS(str_glue("../proc/{job_id}_till_umap.rds"))


################################################################################
# cluster analysis
# pooling cluster tokens
# and calc their freq
clusters <- c(5, 2, 0, 14, 15, 16, 17)

pick_cluster <- select_cluster(cls_so)
clst_list <- map(clusters, pick_cluster)
length(clst_list)

tokens_pooler <- pool_cluster_tokens(input_tokens)
pooled_tokens_list <- map(clst_list, tokens_pooler)
length(pooled_tokens_list)

pooled_freq_list <- map(pooled_tokens_list, table)

freq_dt_list <- map(pooled_freq_list, as.data.table)

freq_dt_list <- imap(freq_dt_list, function(dt, idx) dt[, cluster := clusters[idx]])

freq_dt <- reduce(freq_dt_list, rbind)


# finding the sum of the tokens in a cluster
# to be able to do prop later
# some processing dt functions.
freq_dt[, sum_tokens := sum(.SD$N), by = cluster]
freq_dt[, prop_N := N / sum_tokens]
freq_dt[, prop_labels := round(prop_N, 4)]
freq_dt[, token := as.integer(token)]
setnames(
    freq_dt,
    "V1",
    "token"
)
setkey(freq_dt, token)
freq_dt
################################################################################


################################################################################
# plotting freq as line plot
density_dt <- melt.data.table(freq_dt,
                id.vars = "cluster",
                measure.vars = "token"
)

freq_density_plot <- ggplot(
    data = density_dt,
    aes(x = value,
        #         y = prop_N,
        group = factor(cluster),
        fill = factor(cluster)
    )
) +
         #          geom_line()
         #     stat_smooth(method = "lm", formula = y ~ poly(x, 5))
         geom_density(adjust = 1.5, alpha = 0.4)
#     geom_col()
ggsave(filename = str_glue("../plots/{job_id}_freq_density.pdf"),
       plot = freq_density_plot
)


################################################################################
# take the top n of each cluster
# and do stats on it
top_n <- 50
cutoff_dt <- freq_dt[, .SD[order(-N)][1:top_n], by = cluster]
cutoff_dt[, cluster_rank := rank(prop_N), by = cluster]
cutoff_dt[, token := as.integer(token)]
setkey(cutoff_dt, token)
cutoff_dt


freq_plot <- ggplot(
    data = cutoff_dt,
    aes(x = token,
        y = factor(cluster),
        size = prop_N,
        label = prop_labels,
        color = cluster_rank
    )
) +
    geom_point() +
    #     geom_text(vjust = -2, size = 3, color = "black") +
    scale_color_gradient(low = "lightblue", high = "darkblue") +
    scale_y_discrete(name = "Clusters",
                     limits = factor(clusters)) +
    xlab("Tokens") +
    ggtitle(str_glue("Token Usage among clusters: Top {top_n} tokens")) +
    theme_bw() +
    theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))

ggsave(
    filename = str_glue("../plots/{job_id}_5_24_0_13-15_raw_token_usage_top{top_n}.pdf"),
    plot = freq_plot,
    width = 14,
    height = 7
)
ggsave(
    filename = str_glue("../plots/{job_id}_5_24_0_13-15_prop_token_usage_top{top_n}.pdf"),
    plot = freq_plot,
    width = 20,
    height = 10
)


# getting the sequence of the top_n tokens
cutoff_dt <- vocab_dt[cutoff_dt]
fwrite(x = cutoff_dt,
       file = str_glue("../proc/{job_id}_7_clusters_{top_n}_tokens_sequence.tsv"),
       sep = "\t"
)


################################################################################
# checking the dist of top tokens in the indiv
# tensors. Treating each tensor like a cell
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

################################################################################
# take the bottom n of each cluster
# and do some stats on that
bottom_n <- 10
cutoff_dt <- freq_dt[, .SD[order(N)][1:bottom_n], by = cluster]
cutoff_dt


freq_plot <- ggplot(
    data = cutoff_dt,
    aes(x = token,
        y = factor(cluster),
        size = N,
        label = N,
        color = N
    )
) +
    geom_point() +
    geom_text(vjust = -2, size = 3, color = "black") +
    scale_color_gradient(low = "lightblue", high = "darkblue") +
    scale_y_discrete(name = "Clusters",
                     limits = factor(clusters)) +
    xlab("Tokens") +
    ggtitle(str_glue("Token Usage among clusters: Bottom {bottom_n} tokens")) +
    theme_bw() +
    theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))

ggsave(
    filename = str_glue("../plots/{job_id}_5_24_0_13-15_raw_bottom{bottom_n}_tokens_usage.pdf"),
    plot = freq_plot,
    width = 14,
    height = 7
)

################################################################################
# reading the vocab dict and making a dt out of it
raw_dict <- fromJSON(file = "../proc/bpe.json")
str(raw_dict)
head(raw_vocab) 
raw_vocab[31775]


vocab_dt <- data.table(
    token = names(raw_dict),
    sequence = raw_dict
)
vocab_dt[, sequence := as.character(sequence)]
vocab_dt[1:10]


fwrite(file = "../proc/bpe_vocab.tsv",
       sep = "\t",
       vocab_dt
)

################################################################################
# getting the marker tokens_pooler
# every iter, it reorders the list to put the current idx 
# vec to the top. Then do setdiff in a reduce fashion
unique_tokens <- imap(pooled_tokens_list, function(cluster, idx) {
    union_tokens <- c(pooled_tokens_list[idx], pooled_tokens_list[-idx])
    reduce(union_tokens, setdiff)
    })

# sanity check
str(unique_tokens)
map(unique_tokens, function(x) 30487 %in% x)

unique_tokens_dtlist <- map(unique_tokens, as.data.table)
unique_tokens_dtlist <- imap(unique_tokens_dtlist, function(dt, idx) dt[, cluster := clusters[idx]])
unique_tokens_dt <- reduce(unique_tokens_dtlist, rbind)
setnames(
    unique_tokens_dt,
    "V1",
    "token"
)
setkey(unique_tokens_dt, token)
unique_tokens_dt

fwrite(file = "../proc/unique_tokens_5_24_0_13-15.tsv",
       unique_tokens_dt,
       sep ="\t"
)

# joining
unique_tokens_freq_dt <- freq_dt[unique_tokens_dt, on = .(token, cluster), nomatch = NULL]
unique_tokens_freq_dt

# 365 uniques for cluster 0
unique_tokens_freq_dt[cluster == 0]


freq_plot <- ggplot(
    data = unique_tokens_freq_dt[cluster != 0],
    aes(x = factor(token),
        y = factor(cluster),
        size = N,
        label = N,
        color = N
    )
) +
    geom_point() +
    geom_text(vjust = -2, size = 3, color = "black") +
    scale_color_gradient(low = "lightblue", high = "darkblue") +
    scale_y_discrete(name = "Clusters",
                     limits = factor(clusters[-3])) +
    xlab("Tokens") +
    ggtitle(str_glue("unique Tokens")) +
    theme_bw() +
    theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))

ggsave(
    filename = str_glue("../plots/{job_id}_5_24_0_13-15_unique_tokens.pdf"),
    plot = freq_plot,
    width = 20,
    height = 10
)


unique_tokens_freq_dt <- vocab_dt[unique_tokens_freq_dt, on = .(token), nomatch = NULL]
unique_tokens_freq_dt
fwrite(file = "../proc/unique_tokens_5_24_0_13-15.tsv",
       sep = "\t",
       unique_tokens_freq_dt
)


################################################################################
# get the tensors having the marker tokens
str(cluster_tensors)
str(marker_tokens)
marker_tokens_freq_dt
cluster_tensors[[1]][2, ] %in% marker_tokens[[1]] |> sum()

token_tensor_finder <- find_tensors_with_tokens(marker_tokens_freq_dt, clusters)
marker_tensors <- imap(cluster_tensors, function(mat, idx) token_tensor_finder(mat, idx))
marker_tokens

# sanity check
input_tokens[22087, ] %in% marker_tokens_freq_dt[cluster == clusters[6], token] |> sum()

marker_tokens[[2]]
length(marker_tensors[[2]])


# grab the tensors now
str(input_tokens)
input_tokens[17471, 508:512]
dim(input_tokens[marker_tensors[[6]], ])[1] == length(marker_tensors[[6]])

################################################################################


################################################################################
# analysing marker genes
cls_markers <- fread("../proc/cls_markers.tsv")
cls_markers

# ordering desc by avg_log2FC
cls_markers <- cls_markers[, .SD[order(-avg_log2FC)], by = cluster]
cls_markers

cls_markers[cluster == 5]
cls_markers[gene == "dim-19"
            ][order(-avg_log2FC)]

cls_markers[cluster == 48]

################################################################################
# decoding cluster sequences
str(cluster_tensors)

n_threads <- as.integer(system("nproc", intern = T))
n_threads
plan(multicore, workers = n_threads - 1)
plan()

vocab_hashtable <- hash(keys = vocab_dt$token,
                        values = vocab_dt$sequence
)
vocab_hashtable[[as.character(6)]]


decoder <- decode_tensors(input_tokens, vocab_hashtable)
cluster_seq <- future_map(cluster_tensors,
    function(cluster) apply(cluster, 1, decoder)
)
str(cluster_seq)

saveRDS(cluster_seq,
        file = "../proc/cluster_seq.rds"
)

future_imap(cluster_seq,
           function(cluster, idx) write_fasta(cluster, idx)
)





