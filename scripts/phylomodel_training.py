from typing import Sequence, TypeVar
import pandas as pd
from aux_msa_functions import *
import numpy as np
import torch

from torch.utils.data import Dataset
import pickle as pkl
from select_gpu import get_free_gpu
import torch.nn as nn
import re

from phylomodel_training_aux_fns import *

data_shape = "2D"

if data_shape == "2D":
    from phylomodel_models import *
else:
    from phylomodel_models_3d import *

torch.set_grad_enabled(True)

with open("../data/processed-train-test-data/train_test_sets_PF00004_size_4_no_leak_cls_pp.pkl","rb") as f:
   data = pkl.load(f)

# X_train = data[0]
# X_test = data[1]
# y_train_bl = data[2]
# y_test_bl = data[3]
# y_train_pc = data[4]
# y_test_pc = data[5]

X_train = data["X_train"]
X_test = data["X_test"]
y_train_bl = data["y_train_bl"]
y_test_bl = data["y_test_bl"]
y_train_pc = data["y_train_pc"]
y_test_pc = data["y_test_pc"]

train_dataset = TreeDataset((X_train, y_train_pc))
test_dataset = TreeDataset((X_test, y_test_pc))

criterion = nn.CrossEntropyLoss()
criterion_sum = nn.CrossEntropyLoss(reduction="sum")
criterion_frob = nn.L1Loss(reduction="mean")
gpu = int(get_free_gpu())
device = f"cuda:{gpu}" if torch.cuda.is_available() else "cpu"


checkpoint_path = "./fit-PF00004-size-4-10-epochs.pt"
lamb = 0.1

if checkpoint_path != None:
    checkpoint = torch.load(checkpoint_path, weights_only=True)

    model = ParentPredictor(**checkpoint["model_hparams"]).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])

    optimizer = torch.optim.Adam(model.parameters()) 
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1)
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    cur_epochs = checkpoint['epoch']
    batch_size = checkpoint['batch_size']
    hidden_dim = checkpoint['model_hparams']['hidden_dim']
    embed_dim = checkpoint['model_hparams']['embed_dim']
    n_heads = checkpoint['model_hparams']['n_heads']
    n_layers = checkpoint['model_hparams']['n_layers']
    output_dim = checkpoint['model_hparams']['output_dim']
    input_dim = checkpoint['model_hparams']['input_dim']
    lr = checkpoint['learning_rate']
    num_epochs = 15

else:
    input_dim = 768 ## use 34 for one hot encoding inputs
    lr = 0.0001
    hidden_dim = 64
    embed_dim = 64
    n_heads = 4
    n_layers = 2
    batch_size = 5  ## use batch size 1 when not using internal nodes
    output_dim = 1000
    cur_epochs = 0
    num_epochs = 10

    model = ParentPredictor(input_dim, hidden_dim=hidden_dim, embed_dim=embed_dim, n_heads=n_heads, n_layers=n_layers, output_dim=output_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                        batch_size=batch_size, 
                                        collate_fn=collate_tensors, ## no_int when no internal nodes provided
                                        shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                        batch_size=1, 
                                        collate_fn=collate_tensors,
                                        shuffle=False)

n_total_steps = len(train_loader)

train_epoch_losses = []
test_epoch_losses = []



for epoch in range(cur_epochs, num_epochs):

    train_epoch_outputs = []
    train_labels = []

    model.train()

    for i, (data, labels) in enumerate(train_loader):

        n_train_nodes = 0

        total_loss = torch.tensor(0, dtype=torch.float32).to(device)
        ce_loss = torch.tensor(0, dtype=torch.float32).to(device)
        frob_loss = torch.tensor(0, dtype=torch.float32).to(device)
        regularizer_loss_children = torch.tensor(0, dtype=torch.float32).to(device)
        regularizer_loss_tril = torch.tensor(0, dtype=torch.float32).to(device)

        n_int_nodes_list = []

        for j in range(len(labels)):

            total_nodes = (labels[j] != -1).sum()
            n_int_nodes = int((total_nodes + 1)/2 - 1)

            n_int_nodes_list.append(n_int_nodes) 

        n_int_nodes_list = torch.tensor(n_int_nodes_list)

        for n_nodes in n_int_nodes_list.unique():

            data_j = data[n_int_nodes_list == n_nodes]
            labels_j = labels[n_int_nodes_list == n_nodes]

            msa_index_vector_j = [torch.tensor([ind] * len(labels_j[0])) for ind in range(len(data_j))]
            msa_index_vector_j = torch.concat(msa_index_vector_j)

            seq_mask = (labels_j != torch.tensor(-1)) ## -1 represents padding sequences in the batch
            data_mask = ~((data_j == 0).all(dim=-1)).all(dim=-1) ## use for 3D inputs

            
            data_j = data_j[seq_mask].to(device)
            # data_j = data_j[data_mask].to(device) ## use for 3D inputs
            labels_j = labels_j[seq_mask].to(device)
            msa_index_vector_j = msa_index_vector_j[seq_mask.flatten()].to(device)

            attn_mask = (msa_index_vector_j.unsqueeze(0) == msa_index_vector_j.unsqueeze(1)).float() 
            attn_mask = (1 - attn_mask) * -1e9

            outputs = model(data_j, attn_mask = attn_mask).squeeze(-1)[...,:n_nodes]

            root_mask = labels_j != -2

            cur_ce_loss = criterion_sum(outputs[root_mask, :], labels_j[root_mask])

            # outputs_softmax = torch.softmax(outputs, dim = 1)

            # outputs_tril = outputs_softmax.clone()

            cur_regularizer_loss_children = torch.tensor(0, dtype=torch.float32).to(device)
            cur_regularizer_loss_tril = torch.tensor(0, dtype=torch.float32).to(device)
            cur_frob_loss = torch.tensor(0, dtype=torch.float32).to(device)

            # safe_labels_j = labels_j.clone()
            # safe_labels_j[safe_labels_j < 0] = 0 

            # one_hot_labels = torch.nn.functional.one_hot(safe_labels_j, num_classes = n_nodes).to(torch.float32)
            # one_hot_labels[labels_j < 0] = 0
            
            # for msa_ind in msa_index_vector_j.unique():
                    
            #         msa_mask = msa_index_vector_j == msa_ind
            #         size = msa_mask.sum()
            #         mask = torch.triu(torch.ones(size, size), diagonal=0).bool()[...,:n_nodes].to(device)

                    # assert (one_hot_labels[msa_mask] == one_hot_labels[msa_mask].masked_fill(mask, 0)).all(), "Not lower triangular"
                    # cur_regularizer_loss_children += criterion_frob(outputs_softmax[msa_mask][1:,:].sum(dim = 0), torch.tensor(2).to(device))
                    # outputs_tril[msa_mask] = outputs_tril[msa_mask].masked_fill(mask, 0)
                    # cur_regularizer_loss_tril += criterion_frob(outputs_softmax[msa_mask][1:],outputs_tril[msa_mask][1:])
                    # outputs = outputs.squeeze(0)
                    # outputs[msa_mask] = outputs[msa_mask].masked_fill(mask, float('-inf'))
                    # outputs[msa_mask] = 2 * torch.softmax(outputs[msa_mask], dim = 0)
                    # cur_frob_loss += criterion_frob(outputs[msa_mask][1:], one_hot_labels[msa_mask][1:])
                    # cur_frob_loss += criterion_frob(outputs_softmax[msa_mask][1:], one_hot_labels[msa_mask][1:])
            
            total_loss += cur_ce_loss + (0 * lamb * cur_regularizer_loss_tril) + (0 * lamb * cur_regularizer_loss_children)
            ce_loss += cur_ce_loss

            # total_loss += cur_frob_loss + (1 * lamb * cur_regularizer_loss_tril) + (0 * lamb * cur_regularizer_loss_children)
            # frob_loss += cur_frob_loss 

            regularizer_loss_children += cur_regularizer_loss_children
            regularizer_loss_tril += cur_regularizer_loss_tril
            n_train_nodes += len(labels_j)
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if (i+1) % 1 == 0:

            print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], mean CE Loss: {(ce_loss/n_train_nodes).item():.4f},\
                   mean childreg Loss: {(regularizer_loss_children/n_train_nodes).item():.4f},  mean trilreg Loss: {(regularizer_loss_tril/n_train_nodes).item():.4f}')
            
            # print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], frob loss: {(frob_loss/len(data)).item():.4f},\
            #        mean childreg Loss: {(regularizer_loss_children/n_train_nodes).item():.4f},  mean trilreg Loss: {(regularizer_loss_tril/n_train_nodes).item():.4f}')

    scheduler.step()

    model.eval()

    with torch.no_grad():

        train_epoch_loss = torch.tensor(0, dtype=torch.float32).to(device)
        n_train_nodes = 0
        n_train_trees = 0
        count = 0
       
        for i, (data, labels) in enumerate(train_loader):
            
            n_int_nodes_list = []
            n_train_trees += len(data)

            for j in range(len(labels)):

                total_nodes = (labels[j] != -1).sum()
                n_int_nodes = int((total_nodes + 1)/2 - 1)

                n_int_nodes_list.append(n_int_nodes) 


            n_int_nodes_list = torch.tensor(n_int_nodes_list)

            for n_nodes in n_int_nodes_list.unique():

                data_j = data[n_int_nodes_list == n_nodes]
                labels_j = labels[n_int_nodes_list == n_nodes]

                cur_n_train = len(data_j)

                msa_index_vector_j = [torch.tensor([ind] * len(labels_j[0])) for ind in range(len(data_j))]
                msa_index_vector_j = torch.concat(msa_index_vector_j)

                seq_mask = (labels_j != torch.tensor(-1)) ## -1 represents padding sequences in the batch
                data_mask = ~((data_j == 0).all(dim=-1)).all(dim=-1)

                data_j = data_j[seq_mask].to(device)
                # data_j = data_j[data_mask].to(device) ## use for 3D inputs
                labels_j = labels_j[seq_mask].to(device)
                msa_index_vector_j = msa_index_vector_j[seq_mask.flatten()].to(device)

                attn_mask = (msa_index_vector_j.unsqueeze(0) == msa_index_vector_j.unsqueeze(1)).float() 
                attn_mask = (1 - attn_mask) * -1e9

                outputs = model(data_j, attn_mask = attn_mask).squeeze(-1)[...,:n_nodes]

                # outputs_softmax = torch.softmax(outputs, dim = 1)

                # safe_labels_j = labels_j.clone()
                # safe_labels_j[safe_labels_j < 0] = 0  # e.g. map -2 to 0 just to avoid error

                # one_hot_labels = torch.nn.functional.one_hot(safe_labels_j, num_classes = n_nodes).to(torch.float32)
                # one_hot_labels[labels_j < 0] = 0

                root_mask = labels_j != -2

                # for msa_ind in msa_index_vector_j.unique():

                #     count += 1
                #     msa_mask = msa_index_vector_j == msa_ind
                #     size = msa_mask.sum()
                #     mask = torch.triu(torch.ones(size, size), diagonal=0).bool()[...,:n_nodes].to(device)
                #     outputs[msa_mask] = outputs[msa_mask].masked_fill(mask, float('-inf'))
                #     outputs[msa_mask] = 2 * torch.softmax(outputs[msa_mask], dim = 0)
                #     train_epoch_loss += criterion_frob(outputs[msa_mask][1:, :], one_hot_labels[msa_mask][1:,:])
                    # train_epoch_loss += criterion_frob(outputs_softmax[msa_mask][1:, :], one_hot_labels[msa_mask][1:,:])

                train_epoch_loss += criterion_sum(outputs[root_mask, :], labels_j[root_mask])
                n_train_nodes += len(labels_j[root_mask])

                

        print (f'Epoch [{epoch+1}/{num_epochs}], mean frob Loss (train): {(train_epoch_loss/n_train_nodes).item():.4f}')
        train_epoch_losses.append((train_epoch_loss/n_train_nodes).item())

        # print (f'Epoch [{epoch+1}/{num_epochs}], mean frob Loss (train): {(train_epoch_loss/n_train_trees).item():.4f}')
        # train_epoch_losses.append((train_epoch_loss/n_train_trees).item())

    with torch.no_grad():

        test_epoch_loss = torch.tensor(0, dtype=torch.float32).to(device)
        n_test_nodes = 0
        n_test_trees = 0

        for i, (data, labels) in enumerate(test_loader):

            n_int_nodes_list = []
            n_test_trees += len(data)

            for j in range(len(labels)):

                total_nodes = (labels[j] != -1).sum()
                n_int_nodes = int((total_nodes + 1)/2 - 1)

                n_int_nodes_list.append(n_int_nodes) 


            n_int_nodes_list = torch.tensor(n_int_nodes_list)

            for n_nodes in n_int_nodes_list.unique():

                data_j = data[n_int_nodes_list == n_nodes]
                labels_j = labels[n_int_nodes_list == n_nodes]

                cur_n_test = len(data_j)

                msa_index_vector_j = [torch.tensor([ind] * len(labels_j[0])) for ind in range(len(data_j))]
                msa_index_vector_j = torch.concat(msa_index_vector_j)

                seq_mask = (labels_j != torch.tensor(-1)) ## -1 represents padding sequences in the batch
                data_mask = ~((data_j == 0).all(dim=-1)).all(dim=-1)

                data_j = data_j[seq_mask].to(device)
                # data_j = data_j[data_mask].to(device) ## use for 3D inputs
                labels_j = labels_j[seq_mask].to(device)
                msa_index_vector_j = msa_index_vector_j[seq_mask.flatten()].to(device)

                attn_mask = (msa_index_vector_j.unsqueeze(0) == msa_index_vector_j.unsqueeze(1)).float() 
                attn_mask = (1 - attn_mask) * -1e9

                outputs = model(data_j, attn_mask = attn_mask).squeeze(-1)[...,:n_nodes]

                # outputs_softmax = torch.softmax(outputs, dim = 1)

                # safe_labels_j = labels_j.clone()
                # safe_labels_j[safe_labels_j < 0] = 0  # e.g. map -2 to 0 just to avoid error

                # one_hot_labels = torch.nn.functional.one_hot(safe_labels_j, num_classes = n_nodes).to(torch.float32)
                # one_hot_labels[labels_j < 0] = 0

                root_mask = labels_j != -2

                # for msa_ind in msa_index_vector_j.unique():
                
                #     msa_mask = msa_index_vector_j == msa_ind
                #     size = msa_mask.sum()
                #     mask = torch.triu(torch.ones(size, size), diagonal=0).bool()[...,:n_nodes].to(device)
                #     outputs[msa_mask] = outputs[msa_mask].masked_fill(mask, float('-inf'))
                #     outputs[msa_mask] = 2 * torch.softmax(outputs[msa_mask], dim = 0)
                #     test_epoch_loss += criterion_frob(outputs[msa_mask][1:, :], one_hot_labels[msa_mask][1:,:])
                    # test_epoch_loss += criterion_frob(outputs_softmax[msa_mask][1:, :], one_hot_labels[msa_mask][1:,:])

                test_epoch_loss += criterion_sum(outputs[root_mask, :], labels_j[root_mask])
                n_test_nodes += len(labels_j[root_mask])

        print (f'Epoch [{epoch+1}/{num_epochs}], mean CE Loss (test): {(test_epoch_loss/n_test_nodes).item():.4f}')
        test_epoch_losses.append((test_epoch_loss/n_test_nodes).item())

        # print (f'Epoch [{epoch+1}/{num_epochs}], mean frob Loss (test): {(test_epoch_loss/n_test_trees).item():.4f}')
        # test_epoch_losses.append((test_epoch_loss/n_test_trees).item())

    model_params_dict = {}

    for name, p in model.named_parameters():
        if p.grad is not None:
            model_params_dict[name] = p.grad.abs().mean().item()

    with open(f"reg_grads.pkl", "wb") as f:
            pkl.dump(model_params_dict, f)


    if (epoch + 1) % 20 == 0 or epoch == num_epochs - 1:

        text = f"{epoch + 1}epochs_{lr}lr_{batch_size}batch_under_200"

        with open(f"train_losses_{text}.pkl", "wb") as f:
            pkl.dump(train_epoch_losses, f)

        with open(f"test_losses_{text}.pkl", "wb") as f:
            pkl.dump(test_epoch_losses,f)
            
        torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'model_hparams': {
                    'input_dim': input_dim,
                    'hidden_dim': hidden_dim,
                    'embed_dim': embed_dim,
                    'n_heads':n_heads,
                    'n_layers':n_layers,
                    'output_dim':output_dim
                    },
                    'batch_size': batch_size,
                    'learning_rate':lr
                    }, f"fit-PF00004-size-4-{epoch + 1}-epochs.pt")

