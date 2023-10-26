import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import model



class Train():
    def __init__(self,
                 vocab_size,
                 embed_dim,
                 n_heads,
                 n_layers,
                 ff_dim,
                 max_seq_len,
                 dropout,
                 n_epochs):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.ff_dim = ff_dim
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.n_epochs = n_epochs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # initialising the models
        self.gbert = model.GBERT(self.vocab_size,
                                 self.embed_dim,
                                 self.n_heads,
                                 self.n_layers,
                                 self.ff_dim,
                                 self.max_seq_length,
                                 self.dropout).to(self.device)
        self.mlm_layer = model.MLMLayer(self.vocab_size, self.embed_dim)

        # setting the optimizer and setting the model to train.
        self.optimizer = optim.Adam(self.gbert.parameters(),)
        self.gbert.train()


    def training_cycle(self, data, batch_size):
        data_loader = DataLoader(data, batch_size=batch_size, shuffle=False)


        for epoch in range(self.n_epochs):
            for batch in data_loader:
                input_data = batch[0]
                input_data = input_data.to(self.device)

                # iteration of the model
                output = self.gbert(input_data)
                
                # MLM Preds.
                # implement a class for masking and do the masking here
                mlm_predictions = self.mlm_layer(masked_input_ids)
                mlm_loss = F.cross_entropy(mlm_predictions.view(-1, vocab_size),
                                           input_ids.view(-1),
                                           ignore_index=ignore_index_token)
                mask_for_loss = (masked_input_ids == mask_token_id).float()
                mlm_loss = (mlm_loss * mask_for_loss).sum() / mask_for_loss.sum()

                # Zero the gradients
                self.optimizer.zero_grad()
                # Backpropagate and calculate gradients
                mlm_loss.backward()
                # Update the model parameters
                self.optimizer.step()
                
                # For printing the avg loss for each epoch
                mlm_loss_batch = 0
                mlm_loss_batch = mlm_loss_batch + mlm_loss

        print(f"Epoch: {epoch+1}, Avg_Loss: \n{mlm_loss_batch / len((data_loader))}")

