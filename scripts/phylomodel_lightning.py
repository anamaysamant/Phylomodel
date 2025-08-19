import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl

from typing import Sequence, TypeVar
import pandas as pd
from aux_msa_functions import *
import numpy as np
import torch

from torch.utils.data import Dataset
import pickle as pkl
from phylomodel_models import *
from select_gpu import get_free_gpu
import torch.nn as nn

torch.set_grad_enabled(True)

TensorLike = TypeVar("TensorLike", np.ndarray, torch.Tensor)

class TreeDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, idx):
        X, y = self.data[0][idx], self.data[1][idx]
        return X, y

def collate_tensors(
sequences: Sequence[TensorLike], constant_value=0, dtype=None
) -> TensorLike:
    
    batch_size = len(sequences)

    X_batch, y_batch = zip(*sequences)
    X_batch, y_batch = list(X_batch), list(y_batch)
    
    shape_X = [batch_size] + np.max([mat.shape for mat in X_batch], 0).tolist()
    shape_y = [batch_size] + [shape_X[1]]

    if dtype is None:
        dtype = X_batch[0].dtype

    if isinstance(X_batch[0], np.ndarray):
        X_array = np.full(shape_X, constant_value, dtype=dtype)
    elif isinstance(X_batch[0], torch.Tensor):
        X_array = torch.full(shape_X, constant_value, dtype=dtype)

    if isinstance(y_batch[0], np.ndarray):
        y_array = np.full(shape_y, -1, dtype=y_batch[0][0].dtype)
    elif isinstance(y_batch[0], torch.Tensor):
        y_array = torch.full(shape_y, -1, dtype=y_batch[0][0].dtype)
        
    for arr, mat in zip(X_array, X_batch):
        arrslice = tuple(slice(dim) for dim in mat.shape)
        arr[arrslice] = mat

    for arr, mat in zip(y_array, y_batch):
        arrslice = tuple(slice(dim) for dim in mat.shape)
        arr[arrslice] = mat


    return X_array, y_array

# --------------------------
# 2. LightningModule
# --------------------------


class PhylomodelLightning(pl.LightningModule):
    def __init__(self, lamb = 0.1, Large_D = 1000, hidden_dim = 64, 
                 embed_dim = 64, n_heads = 4, n_layers = 2, lr = 0.001, gamma = 1):
        super().__init__()

        self.save_hyperparameters()

        self.model  = ParentPredictor(Large_D, hidden_dim=hidden_dim, transformer_embed_dim=embed_dim, n_heads=n_heads, n_layers=n_layers, output_dim=Large_D)
        self.loss_fn = nn.CrossEntropyLoss(reduce="sum")

        self.n_train_nodes = []
        self.n_validation_nodes = []

        self.train_epoch_losses = []
        self.val_epoch_losses = []

        self.train_step_losses = []
        self.val_step_losses = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):

        data, labels = batch

        n_train_nodes = 0

        total_loss = torch.tensor(0, dtype=torch.float32)
        ce_loss = torch.tensor(0, dtype=torch.float32)
        regularizer_loss = torch.tensor(0, dtype=torch.float32)

        n_int_nodes_list = []

        for j in range(len(labels)):

            total_nodes = (labels[j] != -1).sum()
            n_int_nodes = int((total_nodes + 1)/2 - 1)

            n_int_nodes_list.append(n_int_nodes) 

        n_int_nodes_vector = [torch.tensor([n_nodes] * len(labels[0])) for n_nodes in n_int_nodes_list]
        n_int_nodes_vector = torch.concat(n_int_nodes_vector)

        n_int_nodes_list = torch.tensor(n_int_nodes_list)

        for n_nodes in n_int_nodes_list.unique():

            data_j = data[n_int_nodes_list == n_nodes]
            labels_j = labels[n_int_nodes_list == n_nodes]

            msa_index_vector_j = [torch.tensor([ind] * len(labels_j[0])) for ind in range(len(data_j))]
            msa_index_vector_j = torch.concat(msa_index_vector_j)

            seq_mask = (labels_j != -1)

            data_j = data_j[seq_mask, :]
            labels_j = labels_j[seq_mask]
            msa_index_vector_j = msa_index_vector_j[seq_mask.flatten()]

            attn_mask = (msa_index_vector_j.unsqueeze(0) == msa_index_vector_j.unsqueeze(1)).float() 
            attn_mask = (1 - attn_mask) * -1e9

            outputs = self.model(data_j, attn_mask = attn_mask).squeeze(-1)[...,:n_nodes]

            root_mask = labels_j != -2

            cur_ce_loss = self.loss_fn(outputs[root_mask, :], labels_j[root_mask])
            # cur_regularizer_loss = lamb * torch.linalg.vector_norm(torch.softmax(outputs, dim = 1).sum(dim = 0) - 2).type_as(cur_ce_loss)

            total_loss += cur_ce_loss # + cur_regularizer_loss
            ce_loss += cur_ce_loss
            # regularizer_loss = cur_regularizer_loss
            n_train_nodes += len(labels_j[root_mask])

            self.train_step_losses.append(ce_loss)
            self.n_train_nodes.append(n_train_nodes)

            self.log("train_loss", ce_loss/n_train_nodes)

        return ce_loss/n_train_nodes

    def validation_step(self, batch, batch_idx):

        data, labels = batch

        test_step_loss = torch.tensor(0, dtype=torch.float32)
        n_test_nodes = 0

        n_int_nodes_list = []

        for j in range(len(labels)):

            total_nodes = (labels[j] != -1).sum()
            n_int_nodes = int((total_nodes + 1)/2 - 1)

            n_int_nodes_list.append(n_int_nodes) 

        n_int_nodes_vector = [torch.tensor([n_nodes] * len(labels[0])) for n_nodes in n_int_nodes_list]
        n_int_nodes_vector = torch.concat(n_int_nodes_vector)

        n_int_nodes_list = torch.tensor(n_int_nodes_list)

        for n_nodes in n_int_nodes_list.unique():

            data_j = data[n_int_nodes_list == n_nodes]
            labels_j = labels[n_int_nodes_list == n_nodes]

            msa_index_vector_j = [torch.tensor([ind] * len(labels_j[0])) for ind in range(len(data_j))]
            msa_index_vector_j = torch.concat(msa_index_vector_j)

            seq_mask = (labels_j != -1)

            data_j = data_j[seq_mask, :]
            labels_j = labels_j[seq_mask]
            msa_index_vector_j = msa_index_vector_j[seq_mask.flatten()]

            attn_mask = (msa_index_vector_j.unsqueeze(0) == msa_index_vector_j.unsqueeze(1)).float() 
            attn_mask = (1 - attn_mask) * -1e9

            outputs = self.model(data_j, attn_mask = attn_mask).squeeze(-1)[...,:n_nodes]

            root_mask = labels_j != -2

            test_step_loss += self.loss_fn(outputs[root_mask, :], labels_j[root_mask])
            n_test_nodes += len(labels_j[root_mask])

        self.test_epoch_outputs.append(test_step_loss)
        self.n_test_nodes.append(n_test_nodes)

        # self.log_dict("val_loss", test_step_loss/n_test_nodes, prog_bar=True)

    def on_train_epoch_end(self):

        mean_train_epoch_loss = np.sum(self.train_step_losses)/np.sum(self.n_train_nodes)
        self.log("mean_training_epoch_loss", mean_train_epoch_loss)

        self.train_step_losses.clear()
        self.n_train_nodes.clear()


    def on_validation_epoch_end(self):

        mean_val_epoch_loss = np.sum(self.validation_step_losses)/np.sum(self.n_validation_nodes)
        self.log("mean_validation_epoch_loss", mean_val_epoch_loss)

        self.train_step_losses.clear()
        self.n_train_nodes.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma= self.hparams.gamma)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss" 
        }


# --------------------------
# 4. Training
# --------------------------
if __name__ == "__main__":

    pl.seed_everything(42)

    Large_D = 768
    hidden_dim = 64
    embed_dim = 256
    n_heads = 4
    n_layers = 2
    batch_size = 50
    lr = 0.001
    num_epochs = 100
    gamma = 1

    with open("../data/train_test_sets_MSA_transf_dirichlet.pkl","rb") as f:
        data = pkl.load(f)

    X_train = data[0]
    X_val = data[1]
    y_train_bl = data[2]
    y_val_bl = data[3]
    y_train_pc = data[4]
    y_val_pc = data[5]

    model = PhylomodelLightning(Large_D=Large_D, hidden_dim=hidden_dim,
                                 embed_dim=embed_dim, n_heads=n_heads, n_layers=n_layers, lr = lr, gamma=gamma)

    train_dataset = TreeDataset((X_train, y_train_pc))
    val_dataset = TreeDataset((X_val, y_val_pc))

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                            batch_size=batch_size, 
                                            collate_fn=collate_tensors,
                                            shuffle=True)

    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, 
                                            batch_size=1, 
                                            collate_fn=collate_tensors,
                                            shuffle=False)


    trainer = pl.Trainer(
        max_epochs=num_epochs,
        accelerator="auto",
        devices="auto",
        precision="16-mixed",
        log_every_n_steps=10
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)