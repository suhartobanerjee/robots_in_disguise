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


pool_cluster_tokens <- function(input_tokens) {

    function(clst_idx) {

        clst_tokens <- input_tokens[clst_idx,]
        pooled_tokens <- as.vector(clst_tokens) |>
            discard(function(x) x < 3)


        return(pooled_tokens)
    }
}
