library(rhdf5)

datasets <- H5Fopen("../../proc/6531478_datasets.h5")
datasets

embed_dim <- datasets$embed_dim
dim(embed_dim)
class(embed_dim)
new_embed_dim <- aperm(embed_dim, c(3, 2, 1))
dim(new_embed_dim)
embed_dim[1, 1, 1]
new_embed_dim[1, 1, 1]

input_tokens <- datasets$input_tokens
dim(input_tokens)
input_tokens <- aperm(input_tokens, c(2, 1))
input_tokens[1, 4]
dim(input_tokens)
