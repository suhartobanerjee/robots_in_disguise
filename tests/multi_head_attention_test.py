import sys
import torch
sys.path.append("../src/")
import model



embed_size = 6
batch_size = 2
seq_length = 3
n_heads = 2

x = torch.rand((batch_size, seq_length, embed_size))
print((x.shape))


attention = model.MultiHeadAttention(embed_size, n_heads)
attention_output = attention.forward(x, x, x)


print((attention_output.shape))
print((attention_output))
