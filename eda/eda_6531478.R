library(data.table)
library(rhdf5)
library(stringr)
library(purrr)
library(Seurat)
library(ggplot2)
library(RColorBrewer)
library(circlize)


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

pick_cluster <- select_cluster(cls_so)
#clst5_idx <- pick_cluster(5)
#clst5_idx[1:5]

clsts <- c(0, 13, 14, 15)
clst_list <- map(clsts, pick_cluster)
length(clst_list)


tokens_pooler <- pool_cluster_tokens(input_tokens)
#pooled_tokens <- tokens_pooler(clst5_idx)

pooled_tokens_list <- map(clst_list, tokens_pooler)
length(pooled_tokens_list)
pooled_tokens_list[[1]][1:15]

# pool all tokens from cluster
length(pooled_tokens) + 2986 * 2 == 1528832

# pooled_freq <- table(pooled_tokens)
pooled_freq[1:6]

pooled_freq_list <- map(pooled_tokens_list, table)
pooled_freq_list[[1]][1]


freq_dt_list <- map(pooled_freq_list, as.data.table)
freq_dt_list

freq_dt_list <- imap(freq_dt_list, function(dt, idx) dt[, cluster := idx])
freq_dt <- reduce(freq_dt_list, rbind)
freq_dt


freq_dt[cluster == 24]
cutoff_dt <- freq_dt[, .SD[order(-N)][1:10], by = cluster]
cutoff_dt

col_pal <- brewer.pal(10, "Purples")
col_pal

freq_plot <- ggplot(
    data = cutoff_dt,
    aes(x = 1,
        y = 1,
        size = N,
        label = V1,
        fill = N
    )
) +
     geom_point(alpha = 0) +
    geom_jitter(shape = 21,
                position = position_jitter(seed = 1)) +
    geom_text(position = position_jitter(seed = 1),
              aes(vjust = -1),
              size = 4
    )+
    scale_fill_gradient(low = "lightgreen", high = "darkgreen") +
    facet_grid(~ cluster)

ggsave(tmp_plot)
ggsave(str_glue("../plots/{job_id}_0_13-15_token_usage.pdf"))
ggsave(str_glue("../plots/{job_id}_5_token_usage.pdf"))


# START HERE
### Plot sectors (outer part)
par(mar=rep(0,4))
circos.clear()
 
### Basic circos graphic parameters
circos.par(cell.padding=c(0,0,0,0), track.margin=c(0,0.15), start.degree = 90, gap.degree =4)
 
### Sector details
circos.initialize(factors = df1$country, xlim = cbind(df1$xmin, df1$xmax))
 
### Plot sectors
circos.trackPlotRegion(ylim = c(0, 1), factors = df1$country, track.height=0.1,
                      #panel.fun for each sector
                      panel.fun = function(x, y) {
                      #select details of current sector
                      name = get.cell.meta.data("sector.index")
                      i = get.cell.meta.data("sector.numeric.index")
                      xlim = get.cell.meta.data("xlim")
                      ylim = get.cell.meta.data("ylim")
 
                      #text direction (dd) and adjusmtents (aa)
                      theta = circlize(mean(xlim), 1.3)[1, 1] %% 360
                      dd <- ifelse(theta < 90 || theta > 270, "clockwise", "reverse.clockwise")
                      aa = c(1, 0.5)
                      if(theta < 90 || theta > 270)  aa = c(0, 0.5)
 
                      #plot country labels
                      circos.text(x=mean(xlim), y=1.7, labels=name, facing = dd, cex=0.6,  adj = aa)
 
                      #plot main sector
                      circos.rect(xleft=xlim[1], ybottom=ylim[1], xright=xlim[2], ytop=ylim[2], 
                                  col = df1$rcol[i], border=df1$rcol[i])
 
                      #blank in part of main sector
                      circos.rect(xleft=xlim[1], ybottom=ylim[1], xright=xlim[2]-rowSums(m)[i], ytop=ylim[1]+0.3, 
                                  col = "white", border = "white")
 
                      #white line all the way around
                      circos.rect(xleft=xlim[1], ybottom=0.3, xright=xlim[2], ytop=0.32, col = "white", border = "white")
 
                      #plot axis
                      circos.axis(labels.cex=0.6, direction = "outside", major.at=seq(from=0,to=floor(df1$xmax)[i],by=5), 
                                  minor.ticks=1, labels.away.percentage = 0.15)
                    })
