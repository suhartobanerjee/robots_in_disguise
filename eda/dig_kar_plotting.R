# some initialisations
job_id <- 6531478
tmp_plot <- "../plots/tmp_plot.pdf"

source("../../digital-karyotype/R/utils.R", chdir = T)


setwd("../../robots_in_disguise/eda/")
cluster_bed <- fread("../proc/seurat_clusters/clusters_7.bed")
# cluster_bed[, cell_name := "cluster_1"]
# cluster_bed[, sv_call_name := "mapping"]
cluster_bed[, unique(sv_state)]


setwd("../../digital-karyotype/R/")
reader_datatable <- initialise_data_reader(input_type = "data_table")
dig_kar_obj <- reader_datatable(
    sv_dt = cluster_bed,
    chr_col = "V1",
    start_col = "V2",
    end_col = "V3",
    cell_col = "V5",
    sv_col = "V7"
)
str(dig_kar_obj)



dig_kar_h1 <- generate_skeleton_h1(dig_kar_obj@ideo_dt)
dig_kar_h1 <- add_centromere(dig_kar_h1, dig_kar_obj@centro_dt)
dig_kar_h1 <- add_gaps_h1(dig_kar_h1, dig_kar_obj@gaps_dt)

col <- RColorBrewer::brewer.pal(7, "Dark2")
names(col) <- paste0("cluster_", c(1:7))
dig_kar_obj@plotting_options@color_arg <- col

dig_kar_h1 <- generate_info_layer_h1(dig_kar_h1,
    dig_kar_obj@data,
    colors_param = dig_kar_obj@plotting_options@color_arg
)
dig_kar_h1 <- stitch_skeleton_h1(dig_kar_h1,
    haplo_label = "Mapping of cluster tensors"
)

setwd("../../robots_in_disguise/eda/")
save_digital_karyotype(
    dig_kar_h1,
    "clusters",
    "clusters_7_woGaps"
)

dig_kar_obj@data <- dig_kar_obj@data[chrom %in% dig_kar_obj@ideo_dt$chrom]
plot_dt <- dig_kar_obj@data[, .N, by = .(sv_state, chrom)]


ggplot(
    data = plot_dt,
    aes(x = chrom,
        y = N,
        fill = sv_state,
    )
) +
    geom_col()

ggsave(file = str_glue("../plots/{job_id}_dist_7clusters_chrom.pdf"),
       width = 10,
       height = 7
)


