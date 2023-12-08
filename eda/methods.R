# Functions to manipulate data

read_datasets <- function(dataset_dir) {
    datasets <- H5Fopen(dataset_dir)

    embed_dim <- datasets$embed_dim
    input_tokens <- datasets$input_tokens
    print("H5 File read. Proceeding to transpose now ...")


    return(list(embed_dim, input_tokens))
}

transpose_matrix <- function(read_result_list) {

    # unpack list
    embed_dim <- read_result_list[[1]]
    input_tokens <- read_result_list[[2]]

    # transpose here
    embed_dim_tr <- aperm(embed_dim, c(3,2,1))
    input_tokens_tr <- aperm(input_tokens, c(2, 1))
    print("Both matrices transposed!")


    return(list(embed_dim_tr, input_tokens_tr))
}


read_labels <- function(label_dir) {

    labels <- fread(str_glue(label_dir),
                    header = F
    )
    labels <- unlist(as.vector(labels))
    names(labels) <- NULL


    return(labels)
}


proc_labels <- function(labels) {
    labels_dt <- data.table(
        idx = c(1:length(labels)),
        labels = labels
    )

    labels_dt[, labels := str_replace_all(labels, "\\[", "")]
    labels_dt[, labels := str_replace_all(labels, "\\]", "")]
    labels_dt[, labels := str_replace_all(labels, "\\'", "")]

    labels_dt[, labels := str_replace_all(labels, "chr23", "chrX")]
    labels_dt[, labels := str_replace_all(labels, "chr24", "chrY")]

    setkey(labels_dt, idx)


    return(labels_dt)
}


extract_tokens <- function(embed_dim) {

    function(token_pos) {
        token_stack <- embed_dim[, token_pos, ]

        return(token_stack)
    }
}


select_cluster <- function(so) {

    function(idx) {

        clst <- subset(so, ident = idx)
        clst_idx <- colnames(clst$RNA@counts)
        clst_idx <- str_replace(clst_idx, "V", "")
        clst_idx <- map_vec(clst_idx, as.integer)


        return(clst_idx)
    }
}


get_cluster_tensors <- function(input_tokens) {

    function(clst_idx) {

        cluster_tensors <- input_tokens[clst_idx,]


        return(cluster_tensors)
    }
}


pool_cluster_tokens <- function(input_tokens) {

    function(clst_idx) {

        get_tensors <- get_cluster_tensors(input_tokens)
        cluster_tensors <- get_tensors(clst_idx)
        pooled_tokens <- as.vector(cluster_tensors) |>
            discard(function(x) x < 3)


        return(pooled_tokens)
    }
}


check_token_tensor <- function(cluster, token) {

    return(cluster %in% token)
}


get_top_tokens_frequency <- function(top_tokens) {

    function(cluster) {
        filtered_cluster <- map(top_tokens, function(token) cluster %in% token)

        # reshape the flattened vec to matrix
        filtered_cluster <- map(filtered_cluster,
                                function(x) matrix(x,
                                                   nrow = dim(cluster)[1],
                                                   ncol = dim(cluster)[2]
                                                   )
        )
        filtered_cluster_dt <- map(filtered_cluster, as.data.table)

        occurence_dt <- imap(filtered_cluster_dt,
                             function(dt, idx) {
                                 occurence_dt <- data.table(
                                    token = top_tokens[idx],
                                    freq = rowSums(dt)
                                 )
                             }
        )
        occurence_dt <- reduce(occurence_dt, rbind)


        return(occurence_dt)
    }
} 


find_tensors_with_tokens <- function(marker_tokens_dt, clusters) {

    function(mat, idx) {
        return(
            which(
                apply(cluster_tensors[[idx]],
                      MARGIN = 1,
                      FUN = function(x) sum(marker_tokens_dt[cluster == clusters[idx],
                                                             token] %in% x)
                      ) > 0
            )
        )
    }
}


decode_tensors <- function(input_tokens, vocab_hashtable) {

    function(tensor) {
        # removing special tokens: cls, sep and chr tokens
        proc_tensor <- discard(tensor, function(x) x < 3) |>
            discard(function(x) x >32000)

        # decoding each token using map
        # then pasting all of them together using reduce
        sequence <- reduce(
            future_map(proc_tensor,
                function(curr_token) vocab_hashtable[[as.character(curr_token)]]
            ),
            paste0
        )


        return(sequence)
    }
}


write_fasta <- function(cluster, cluster_idx) {

    imap(cluster,
        function(tensor, idx) {
            # write the header
            write(
                file = str_glue("../proc/seurat_clusters/cluster_{cluster_idx}.fasta"),
                x = str_glue(">{idx}"),
                sep = "\n",
                append = TRUE
            )
            # write the tensor
            write(
                file = str_glue("../proc/seurat_clusters/cluster_{cluster_idx}.fasta"),
                x = tensor,
                sep = "\n",
                append = TRUE
            )
        }
    )
}
