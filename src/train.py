import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
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
                 n_epochs,
                 batch_size):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.ff_dim = ff_dim
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.n_epochs = n_epochs
        self.batch_size = batch_size

        # class specific data
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.enc_output_list = []
        self.mlm_loss_batch = 0

        # initialising the models
        self.gbert = model.GGBERT(self.vocab_size,
                                 self.embed_dim,
                                 self.n_heads,
                                 self.n_layers,
                                 self.ff_dim,
                                 self.max_seq_len,
                                 self.dropout)
        self.gbert = nn.DataParallel(self.gbert, device_ids=[0,1])
        self.mlm_layer = model.MLMLayer(self.vocab_size, self.embed_dim)

        # setting the optimizer and setting the model to train.
        self.optimizer = optim.Adam(self.gbert.parameters(), lr = 0.5)
        self.loss_func = nn.CrossEntropyLoss(ignore_index = -100)
        self.gbert.train()


    def training_cycle(self, data):
        assert torch.cuda.is_available(), "Cannot see CUDA devices. Please check"
        print(f"CUDA devices : {self.device}")


        for epoch in range(self.n_epochs):
            data_loader = DataLoader(data, batch_size = self.batch_size, shuffle = True)
            print(len(data_loader))
                


            for batch in data_loader:
                input_data = batch[0]
                print(input_data.size())
                input_data = input_data.unsqueeze(0).to(self.device)

                # iteration of the model
                output = self.gbert.to(self.device).forward(input_data)
                self.enc_output_list.append((output))
                
                # MLM Preds.
                # masking, feeding through MLM layer and calc loss
                mask = model.CreateMask()
                masked_input_ids, target_labels = mask.add_mask_token(input_data)
                mlm_predictions, pred_labels = self.mlm_layer(masked_input_ids.to(self.device)).to(self.device)
                mlm_loss = self.loss_func(mlm_predictions.view(-1, self.vocab_size),
                                           target_labels.view(-1))

                # Zero the gradients
                self.optimizer.zero_grad()
                # Backpropagate and calculate gradients
                mlm_loss.backward()
                # Update the model parameters
                self.optimizer.step()
                
                # summing up losses for all batches in the epoch
                self.mlm_loss_batch += mlm_loss.item()


            print(f"Epoch: {epoch+1}, Avg_Loss: {self.mlm_loss_batch / len((data_loader))}")
            self.mlm_loss_batch = 0

        # finally return the enc outputs of all the outputs.
        return self.enc_output_list
