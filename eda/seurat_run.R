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



# fake scaling to continue with the rest of the analysis
cls_so <- ScaleData(cls_so,
                    do.scale = F,
                    do.center = F,
                    features = dims
)
cls_so@assays$RNA@scale.data[1:5,1:5]


# pca
cls_so <- RunPCA(cls_so, npcs = 64, features = dims)

pca_plot <- DimPlot(cls_so, reduction = "pca") + NoLegend()
ggsave(str_glue("../plots/{job_id}_pca.pdf"))

elbw_plot <- ElbowPlot(cls_so, ndims = 64)
ggsave(str_glue("../plots/{job_id}_elbow_plot.pdf"))


# finding clusters
cls_so <- FindNeighbors(cls_so, dims = 1:18)
cls_so <- FindClusters(cls_so, resolution = 0.5)
head(Idents(cls_so), 5)


# umap
cls_so <- RunUMAP(cls_so, dims = 1:18)
umap_plot <- DimPlot(cls_so, reduction = "umap", label = T)
ggsave(str_glue("../plots/{job_id}_umap_0.5.pdf"))


# finding marker dimensions
cls_markers <- FindAllMarkers(cls_so)
cls_markers <- as.data.table(cls_markers)
cls_markers

fwrite(file = "../proc/cls_markers.tsv",
       sep = "\t",
       cls_markers
)


# save rds
saveRDS(cls_so, file = str_glue("../proc/{job_id}_till_umap.rds"))


