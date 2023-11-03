import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import model
import logging



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
        self.mlm_loss_batch = 0

        # initialising the models
        self.gbert = model.GGBERT(self.vocab_size,
                                 self.embed_dim,
                                 self.n_heads,
                                 self.n_layers,
                                 self.ff_dim,
                                 self.max_seq_len,
                                 self.dropout)
        self.n_gpus = torch.cuda.device_count()
        self.gbert = nn.DataParallel(self.gbert,
                                     device_ids = [x for x in range(0, self.n_gpus)]).to(self.device)
        self.mlm_layer = model.MLMLayer(self.vocab_size, self.embed_dim)
        self.mlm_layer = nn.DataParallel(self.mlm_layer,
                                         device_ids = [x for x in range(0, self.n_gpus)]).to(self.device)

        # setting the optimizer and setting the model to train.
        self.optimizer = optim.Adam([{'params': self.gbert.parameters()},
                                     {'params': self.mlm_layer.parameters()}], lr = 0.003)
        self.loss_func = nn.CrossEntropyLoss(ignore_index = -100)
        self.gbert.train()

    def training_cycle(self, data):
        assert torch.cuda.is_available(), "Cannot see CUDA devices. Please check"

        logging.info(f"Device : {self.device}")
        logging.info(f"Number of CUDA devices : {self.n_gpus}")

        # batching the data
        loader = DataLoader(data, batch_size = self.batch_size, shuffle = False)


        for epoch in range(self.n_epochs):

            for batch in loader:
                input_data = batch
                input_data = input_data.to(self.device)

                # iteration of the model
                enc_output = self.gbert.forward(input_data)
                
                # MLM Preds.
                # masking, feeding through MLM layer and calc loss
                mask = model.CreateMask()
                masked_input_ids, target_labels = mask.add_mask_token(input_data)
                mlm_predictions, pred_labels = self.mlm_layer(enc_output)
                mlm_loss = self.loss_func(mlm_predictions.view(-1, self.vocab_size),
                                           target_labels.view(-1))
                logging.debug(f"The original labels : \n{masked_input_ids}\nThe pred_labels are : \n{pred_labels}")

                # Zero the gradients
                self.optimizer.zero_grad()
                # Backpropagate and calculate gradients
                mlm_loss.backward()
                # Update the model parameters
                self.optimizer.step()
                
                # summing up losses for all batches in the epoch
                self.mlm_loss_batch += mlm_loss.item()


            logging.info(f"Epoch: {epoch+1}, Avg_Loss: {self.mlm_loss_batch / len(loader)}")
            self.mlm_loss_batch = 0

        # finally return the enc outputs of all the outputs.
        return enc_output
