library(BSgenome)
library(Biostrings)
library(data.table)
library(stringr)
library(BSgenome.Hsapiens.UCSC.hg38)
library(purrr)
library(itertools)



source("../../digital-karyotype/R/utils.R", chdir = T)


################################################################################
tall_data <- fread("../data/tall_data/TALL03-DEA5.lenient.filtered.txt")
tall_data

reader <- initialise_data_reader("data_table")
dig_kar_obj <- reader("../data/tall_data/TALL03-DEA5.lenient.filtered.txt")
dig_kar_obj
dig_kar_obj@data[str_detect(sv_state, "h2")]


# taking only the unique sv segments
sv_only <- unique(tall_data[, .(chrom, start, end, sv_call_name)])
setnames(
    sv_only,
    c("start", "end", "sv_call_name"),
    c("start_loc", "end_loc", "sv_state")
)


# removing haplo info
sv_only[, haplotype := str_split_i(sv_state, "_", 2)]
sv_only[, sv_state := str_split_i(sv_state, "_", 1)]


# removing complex calls and idup calls for the moment
sv_only <- sv_only[sv_state != "complex"]
sv_only <- sv_only[sv_state != "idup"]
str(sv_only)

################################################################################


################################################################################
# sanity check for ct cells
dig_kar_h2 <- generate_skeleton_h2(dig_kar_obj@ideo_dt)
dig_kar_h2 <- add_centromere(dig_kar_h2, dig_kar_obj@centro_dt)
dig_kar_h2 <- add_gaps_h2(dig_kar_h2, dig_kar_obj@gaps_dt)
dig_kar_h2 <- generate_info_layer_h2(dig_kar_h2, 
                                     dig_kar_obj@data[cell_name == "TALL3x1_DEA5_PE20414" &
                                                      str_detect(sv_state, "h2")],
                                     colors_param = dig_kar_obj@plotting_options@color_arg
)
dig_kar_h2 <- stitch_skeleton_h2(dig_kar_h2)

save_digital_karyotype(
    dig_kar_h2,
    "digital-karyotype",
    "ct_cell_check"
)

################################################################################


################################################################################
# rough section
sv_only
idx <- 2531
seq <- as.character(getSeq(Hsapiens,
       names = "chr2",
       start = 10000,
       end = 10100
))
seq <- getSeq(Hsapiens,
       names = sv_only[idx, chrom],
       start = sv_only[idx, start_loc],
       end = sv_only[idx, end_loc],
       as.character = T
)
length(seq)
str(seq)
str_sub(seq, 1, 5)
class(seq)

iter_obj <- iter(sv_only, by = "row") 
iter_obj
map(iter_obj, function(x) print(x))
nextElem(iter_obj)

################################################################################
# mutating sequence in silico
# getting the original_sequence and mutating accordingly
sv_only[, .(chrom, start_loc, end_loc, sv_state)]

# getting original ref sequences
sv_only[, original_sequence := get_seq_for_sv(.SD)]
str(sv_only)


# func to get the original_sequence from ref
# iterating through the rows and applying this func
# for every row
get_seq_for_sv <- function(sv_only_slice) {
    
    seq <- as.character(getSeq(Hsapiens,
                  names = sv_only_slice[1, chrom],
                  start = sv_only_slice[1, start_loc],
                  end = sv_only_slice[1, end_loc]
                )
    )
    names(seq) <- NULL

    return(seq)

}

########################################
# mutation begins here

# inversion of sequences
sv_only[str_detect(sv_state, "inv"), 
        mutated_sequence := stringi::stri_reverse(original_sequence)]
str(sv_only)


# duplicating the sequence
sv_only[str_detect(sv_state, "^dup"),
        mutated_sequence := str_dup(original_sequence, times = 2)] 
str(sv_only)


# deletion of sequences
sv_only[str_detect(sv_state, "del"),
        mutated_sequence := ""]
sv_only[str_detect(sv_state, "del"), .(chrom, start_loc, end_loc, sv_state, mutated_sequence)]
str(sv_only)

########################################

# removing idup for the time being
sv_only[sv_state != "idup" & chrom == "chr6", .(chrom, start_loc, end_loc, sv_state)]
sv_only <- sv_only[!is.na(mutated_sequence)]
sv_only[is.na(mutated_sequence)]

str(sv_only)

################################################################################


################################################################################
# removing duplicated calls
# basically removing the layer of haplotype info
test_dt <- sv_only[, .(start_loc,
            end_loc,
            sv_state,
            str_sub(mutated_sequence, 1, 5)),
        by = chrom]
test_dt

sv_only_dup_removed <- sv_only[, .SD[!duplicated(.SD$mutated_sequence)],
                               by = .(chrom, start_loc, end_loc)]
str(sv_only_dup_removed)

################################################################################


################################################################################
# removing original_sequence and saving to file
sv_only_save_dt <- copy(sv_only_dup_removed)
sv_only_save_dt[, original_sequence := NULL]
str(sv_only_save_dt)
str(sv_only_dup_removed)

setDTthreads(4)
getDTthreads()
fwrite(file = "../proc/tall_ct_mutated_seq_amp_del_inv.tsv.gz",
       sep = "\t",
       x = sv_only_save_dt,
       compress = "gzip"
)

################################################################################


