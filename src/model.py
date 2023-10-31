import torch
import torch.nn as nn
import math
import copy




class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)
        self.W_o = nn.Linear(embed_dim, embed_dim)
        

    def scaled_dot_product_attention(self, Q, K, V):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
#        if mask is not None:
#            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output
        

    def split_heads(self, x):
        batch_size, seq_length, embed_dim = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        

    def combine_heads(self, x):
        batch_size, _, seq_length, head_dim = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.embed_dim)
        

    def forward(self, Q, K, V):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        
        attn_output = self.scaled_dot_product_attention(Q, K, V)
        output = self.W_o(self.combine_heads(attn_output))
        return output



class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_seq_length):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_seq_length, embed_dim)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]



class EncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        
    def forward(self, x):
        attn_output = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x




class MLMLayer(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super(MLMLayer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pred_layer = nn.Linear(embed_dim, vocab_size)

    def forward(self, masked_input):
        # Apply the embedding layer
        embedded = self.embedding(masked_input)
        # Apply the linear layer to predict token probabilities
        preds = self.pred_layer(embedded)
        pred_label = torch.argmax(preds, dim=-1)


        return (preds, pred_label)




class GGBERT(nn.Module):
    def __init__(self, 
                 vocab_size,
                 embed_dim, num_heads,
                 num_layers,
                 ff_dim,
                 max_seq_length,
                 dropout):
        super(GGBERT, self).__init__()
        self.encoder_embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_encoding = PositionalEncoding(embed_dim, max_seq_length)

        self.encoder_layers = nn.ModuleList([EncoderLayer(embed_dim, num_heads, ff_dim, dropout) for _ in range(num_layers)])

        self.dropout = nn.Dropout(dropout)


    def forward(self, src):
        src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src)))

        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output)


        return enc_output



class CreateMask:
    def __init__(self, prob = 0.15):
        self.prob = prob
        self.tokens_to_exclude = (1, 2, 104)

    def add_mask_token(self, input_data):
        # Create a binary mask where 1 indicates masking and 0 indicates not masking
        # The mask is sampled based on the mask probability
        # excluding certain idx
        mask = torch.bernoulli(torch.full(input_data.shape, self.prob)).bool()

        # Set the masking probability to False for tokens to be excluded
        torch.where(mask[input_data == self.tokens_to_exclude], False, input_data)
        
        masked_input_data = torch.where(mask, torch.tensor(103), input_data)


        # target labels will be of same shape of input_data
        # just the mask loci will be original labels
        # everything else will be -100
        target_labels = torch.where(mask, input_data, torch.tensor(-100))



        return (masked_input_data, target_labels)

