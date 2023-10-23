import sys
sys.path.append(("../src/"))
import model
import torch
import torch.optim as optim


src_vocab_size = 10
embed_dim = 4
num_heads = 2
num_layers = 1
ff_dim = 10
max_seq_length = 10
dropout = 0.1

gbert = model.GBERT(src_vocab_size, embed_dim, num_heads, num_layers, ff_dim, max_seq_length, dropout)

# Generate random sample data
src_data = torch.randint(1, src_vocab_size, (2, max_seq_length))  # (batch_size, seq_length)
print((src_data))



optimizer = optim.Adam(gbert.parameters(),)
gbert.train()


for epoch in range(2):
    optimizer.zero_grad()
    output = gbert(src_data)
    optimizer.step()
    print(f"Epoch: {epoch+1}, Output: \n{output}")
