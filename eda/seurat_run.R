################################################################################
########################################
# cls stack here
cls_so <- CreateSeuratObject(counts = cls_stack_df)
cls_so
cls_so$chrom <- labels_dt$labels
# cls_so@assays$RNA@counts[1:5, 1:5]


# cls_so <- NormalizeData(cls_so, normalization.method = "LogNormalize", scale.factor = 10000)


# Identify the 10 most highly variable genes
cls_so <- FindVariableFeatures(cls_so, selection.method = "vst", nfeatures = 64)
VariableFeatures(cls_so)
top10 <- head(VariableFeatures(cls_so), 10)
top10

# plot variable features with and without labels
plot1 <- VariableFeaturePlot(cls_so)
plot1 <- LabelPoints(plot = plot1, points = top10, repel = TRUE)
plot1 <- plot1 + ggtitle("Highly variable dimensions")
ggsave(str_glue("../plots/{job_id}_cls_highly_variable_dimensions.pdf"))



# fake scaling to continue with the rest of the analysis
cls_so <- ScaleData(cls_so,
    do.scale = F,
    do.center = F,
    features = dims
)
cls_so@assays$RNA@scale.data[1:5, 1:5]


# pca
cls_so <- RunPCA(cls_so, npcs = 64, features = dims)

pca_plot <- DimPlot(cls_so, reduction = "pca") + NoLegend()
ggsave(str_glue("../plots/{job_id}_cls_pca.pdf"))

elbw_plot <- ElbowPlot(cls_so, ndims = 63)
ggsave(str_glue("../plots/{job_id}_cls_elbow_plot.pdf"))


# finding clusters
ndims <- 63
cls_so <- FindNeighbors(cls_so, dims = 1:ndims)
cls_so <- FindClusters(cls_so, resolution = 0.5)
head(Idents(cls_so), 5)


# umap
cls_so <- RunUMAP(cls_so, dims = 1:ndims)
umap_plot <- DimPlot(cls_so, reduction = "umap", label = T) + NoLegend()
ggsave(str_glue("../plots/{job_id}_cls_umap_0.5.pdf"))

umap_plot <- DimPlot(cls_so,
    reduction = "umap",
    group.by = "chrom"
)
ggsave(
    filename = str_glue("../plots/{job_id}_cls_umap_0.5_chrom.pdf"),
    plot = umap_plot,
    width = 8,
    height = 7
)


# finding marker dimensions
cls_markers <- FindAllMarkers(cls_so)
cls_markers <- as.data.table(cls_markers)
cls_markers

fwrite(
    file = "../proc/cls_markers.tsv",
    sep = "\t",
    cls_markers
)


# save rds
saveRDS(cls_so, file = str_glue("../proc/{job_id}_till_umap.rds"))


########################################
# chr stack here
chr_so <- CreateSeuratObject(counts = chr_stack_df)
chr_so
chr_so@assays$RNA@counts[1:5, 1:5]


# chr_so <- NormalizeData(chr_so, normalization.method = "LogNormalize", scale.factor = 10000)


# Identify the 10 most highly variable genes
chr_so <- FindVariableFeatures(chr_so, selection.method = "vst", nfeatures = 64)
VariableFeatures(chr_so)
top10 <- head(VariableFeatures(chr_so), 10)
top10

# plot variable features with and without labels
plot1 <- VariableFeaturePlot(chr_so)
plot1 <- LabelPoints(plot = plot1, points = top10, repel = TRUE)
plot1 <- plot1 + ggtitle("Highly variable dimensions")
ggsave(str_glue("../plots/{job_id}_chr_highly_variable_dimensions.pdf"))



# fake scaling to continue with the rest of the analysis
chr_so <- ScaleData(chr_so,
    do.scale = F,
    do.center = F,
    features = dims
)
chr_so$chrom <- labels_dt$labels
chr_so@assays$RNA@scale.data[1:5, 1:5]


# pca
chr_so <- RunPCA(chr_so, npcs = 64, features = dims)

pca_plot <- DimPlot(chr_so, reduction = "pca") + NoLegend()
ggsave(str_glue("../plots/{job_id}_chr_pca.pdf"))

elbw_plot <- ElbowPlot(chr_so, ndims = 64)
ggsave(str_glue("../plots/{job_id}_chr_elbow_plot.pdf"))


# finding clusters
ndims <- 22
chr_so <- FindNeighbors(chr_so, dims = 1:ndims)
chr_so <- FindClusters(chr_so, resolution = 0.5)
head(Idents(chr_so), 5)


# umap
chr_so <- RunUMAP(chr_so, dims = 1:ndims)
umap_plot <- DimPlot(chr_so, reduction = "umap", label = T) + NoLegend()
ggsave(
    filename = str_glue("../plots/{job_id}_chr_umap_0.5.pdf"),
    plot = umap_plot,
    width = 7,
    height = 7
)

# colouring by chrom
umap_plot <- DimPlot(chr_so,
    reduction = "umap",
    label = T,
    group.by = "chrom"
) + NoLegend()
ggsave(str_glue("../plots/{job_id}_chr_umap_0.5_chrom.pdf"))


# finding marker dimensions
chr_markers <- FindAllMarkers(chr_so)
chr_markers <- as.data.table(chr_markers)
chr_markers

fwrite(
    file = "../proc/chr_markers.tsv",
    sep = "\t",
    chr_markers
)


# save rds
saveRDS(chr_so, file = str_glue("../proc/{job_id}_chr_till_umap.rds"))
