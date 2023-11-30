library(data.table)
library(rhdf5)
library(stringr)
library(purrr)
library(Seurat)
library(ggplot2)


source("./methods.R")

# some initialisations
job_id <- 6531478
chr_names <- "all_chr"
tmp_plot <- "../plots/tmp_plot.pdf"


# reading in the matrices
result_list <- transpose_matrix(
    read_datasets(str_glue("../proc/{job_id}_datasets.h5"))
)
embed_dim <- result_list[[1]]
input_tokens <- result_list[[2]]
dim(embed_dim)
dim(input_tokens)


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
cls_stack[1:2, 1:5]
class(cls_stack)


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
cls_stack_dt[, 1:5]
table(is.na(cls_stack_dt))


cls_stack_df <- as.data.frame(cls_stack_dt)
cls_stack_df[1:5,1:5]
rownames(cls_stack_df) <- cls_stack_df$idx
cls_stack_df$idx <- NULL
cls_stack_df[1:5,1:5]


cls_so <- CreateSeuratObject(counts = cls_stack_df)
cls_so
cls_so@assays$RNA@counts[1:5, 1:5]


cls_so <- NormalizeData(cls_so, normalization.method = "LogNormalize", scale.factor = 10000)


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



vector(cls_stack_dt$idx, mode = "character")
cls_so <- ScaleData(cls_so,
                    do.scale = F,
                    do.center = F,
                    features = dims
)


cls_so@assays$RNA@scale.data[1:5,1:5]


cls_so <- RunPCA(cls_so, npcs = 64, features = dims)

pca_plot <- DimPlot(cls_so, reduction = "pca") + NoLegend()
ggsave(str_glue("../plots/{job_id}_pca.pdf"))

elbw_plot <- ElbowPlot(cls_so)
ggsave(str_glue("../plots/{job_id}_elbow_plot.pdf"))


cls_so <- FindNeighbors(cls_so, dims = 1:18)
cls_so <- FindClusters(cls_so, resolution = 0.5)
head(Idents(cls_so), 5)


saveRDS(cls_so, file = str_glue("./{job_id}_cluster_step.rds"))
cls_so <- readRDS(str_glue("./{job_id}_cluster_step.rds"))


cls_so <- RunUMAP(cls_so, dims = 1:18)
umap_plot <- DimPlot(cls_so, reduction = "umap")
ggsave("../plots/{job_id}_umap_0.5.pdf")


clst5 <- subset(cls_so, ident = 5)
clst5_idx <- colnames(clst5$RNA@counts)
clst5_idx <- str_replace(clst5_idx, "V", "")
clst5_idx <- map_vec(clst5_idx, as.integer)
clst5_idx[1:5]


dim(input_tokens)
clst5_tokens <- input_tokens[clst5_idx,]
dim(clst5_tokens)
clst5_tokens[1,]

max_5 <- max(table(clst5_tokens[1,]))
check <- map_vec(table(clst5_tokens[1,]), function(x) x / max_5)
keep(check, function(x) x > 0.5)




