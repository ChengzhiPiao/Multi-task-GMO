import torch
from tqdm import tqdm
from torch_geometric.nn.models import GIN, GAT, GraphSAGE
import pickle
import pandas as pd
from _revision_result_stability.swapped_column_cell.graph import *
from torch import nn
from sklearn.model_selection import train_test_split
from typing import Optional
from torch_geometric.data import Data, Dataset, Batch
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
import numpy as np
import random
import time
import torch.optim.lr_scheduler as lr_scheduler
import wandb
import warnings
from _revision_result_stability.swapped_column_cell.my_constants import *
from torch_geometric.nn import global_mean_pool

def set_seed() -> None:
    """This functions set the seeds for various libraries

    Args:
        seed (int): this number will be used as seed
    """
    seed = SEED
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class GraphTriplesDataset(Dataset):
    
    def __init__(self, triples: pd.DataFrame, graphs: dict, tables: Optional[dict] = None) -> None:
        """Init function

        Args:
            triples (pd.DataFrame): Dataframe that contains triples ()'r_id','s_id','table_overlap')
            graphs (dict): a dictionary containing a graph for every key that appears in the triples dataset
            tables (Optional[dict], optional): not implemented. Defaults to None.
        """
        super(GraphTriplesDataset, self).__init__()
        self.triples = triples
        self.graphs = graphs

    def len(self) -> int:
        return len(self.triples)
    
    def get(self, idx:int) -> tuple:
        t = self.triples.iloc[idx][:]
        try:
            g1 = self.graphs[str(t.loc['r_id'])]
            g2 = self.graphs[str(t.loc['s_id'])]
        except:
            g1 = self.graphs[str(int(t.loc['r_id']))]
            g2 = self.graphs[str(int(t.loc['s_id']))]
        return Data(g1.X, g1.edges), Data(g2.X, g2.edges), t.loc['a%'], g1.next_column_index, g1.next_row_index, g1.X.shape[0], g2.next_column_index, g2.next_row_index, g2.X.shape[0]
    
    
def train_test_valid_split(df: pd.DataFrame, ttv_ratio: set=(0.8,0.1,0.1)) -> set:
    """Given a dataframe it is divided in a train dataset, a test dataset, and a validation dataset

    Args:
        df (pd.DataFrame): the dataset to split
        ttv_ratio (set, optional): A triple that tells the function how to split the dataset (TRAIN, TEST, VALIDATE). Defaults to (0.8,0.1,0.1).

    Returns:
        set: a set of 3 elements containing the triple split of the dataset in (TRAIN, TEST, VALIDATE)
    """
    train_data, test_valid_data = train_test_split(df, test_size=ttv_ratio[1]+ttv_ratio[2], random_state=SEED)
    test_data, valid_data = train_test_split(   test_valid_data, 
                                                test_size=ttv_ratio[2]/(ttv_ratio[1]+ttv_ratio[2]), 
                                                random_state=SEED)
    return train_data, test_data, valid_data
    

class Armadillo(nn.Module):
    def init_weights(self):
        """uniform initialization for the weights in the network
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def __init__(self, hidden_channels: int=300, num_layers: int=3, dropout: float=0, act: str="relu", 
                 gnn_type: str='GraphSAGE', initial_embedding_method: str='fasttext', model_file: str=None, relu: bool=False) -> None:
        """The init function

        Args:
            hidden_channels (int): size of the generated embeddings
            num_layers (int): number of layers of the network, every embedding will be generated using using his neighbours at distance num_layers
            dropout (float, optional): dropout probability for the weights. Defaults to 0.
            act (str, optional): The activation function between the layers. Defaults to "relu".
            gnn_type (str): the gnn to use, accepted 'GIN', 'GAT', and 'GraphSAGE'. Defaults to 'GIN'
            initial_embedding_method (str, optional): describes the method used to generate the initial embeddings of the nodes in the graph, it determines the number of in_channels. Accepted values are 'fasttext', 'BERT', and 'sha256'. Defaults to 'fasttext'.
            model_file (str, optional): this parameter can be used to load a pretrained model from a model_file
            relu (bool, optional): if set to Tre a relu layer will be added at the end of the network, it will prevent negative cosine similarities between the embeddings
        """
        super(Armadillo,self).__init__()
        if model_file != None:
            state_dict = torch.load(model_file)
            self.hidden_channels = state_dict['hidden_channels']
            self.num_layers = state_dict['num_layers']
            self.dropout = state_dict['dropout']
            self.act = state_dict['act']
            self.gnn_type = state_dict['gnn_type']
            self.in_channels = state_dict['in_channels']
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.hidden_channels = hidden_channels
            self.num_layers = num_layers
            self.dropout = dropout
            self.act = act
            self.gnn_type = gnn_type
            if initial_embedding_method == 'fasttext':
                self.in_channels = 300
            elif initial_embedding_method == 'BERT':
                self.in_channels = 768
            elif initial_embedding_method == 'sha256':
                self.in_channels = 32
            else:
                raise NotImplementedError
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if self.gnn_type == 'GIN':
            self.model = GIN(self.in_channels, self.hidden_channels, self.num_layers, dropout=dropout, act=act).to(self.device)
        elif self.gnn_type == 'GAT':
            self.model = GAT(self.in_channels, self.hidden_channels, self.num_layers, dropout=dropout, act=act).to(self.device)
        elif self.gnn_type == 'GraphSAGE':
            self.model = GraphSAGE(self.in_channels, self.hidden_channels, self.num_layers, dropout=dropout, act=act).to(self.device)
        else:
            raise NotImplementedError
        if relu:
            self.relu = nn.ReLU()   #Delete if you underperform
        else:
            self.relu = None
        if model_file != None:
            self.load_state_dict(state_dict['model_state_dict'])

    def forward(self, b: Batch, start: Batch, end:Batch, size:Batch) -> torch.tensor:
        """Provided a batch of graphs as input their embeddings are provided in output

        Args:
            b (Batch): a group of graphs (X tensor and edgelist tensor)

        Returns:
            torch.tensor: a tensor composed by <num_graphs_int_the_batch> rows, every row will be the embedding of the corresponding graph
        """
        out_gin = self.model(b['x'], b['edge_index'])
        
        offset = 0
        for i in range(len(size)):
            id_start = offset + start[i]
            id_end = offset + end[i]
            end_cell = offset + size[i]
            try:
                vals = out_gin[id_end:end_cell]
                cols = out_gin[offset:id_start]
                try:
                    tmp = torch.cat((cols, vals), dim=0).mean(axis=0).unsqueeze(0)
                except:
                    tmp = torch.cat((cols.unsqueeze(0), vals.unsqueeze(0)), dim=0).mean(axis=0).unsqueeze(0)
                out = torch.cat((out, tmp), dim=0)
            except:
                vals = out_gin[id_end:end_cell]
                cols = out_gin[offset:id_start]
                try:
                    tmp = torch.cat((cols, vals), dim=0).mean(axis=0).unsqueeze(0)
                except:
                    tmp = torch.cat((cols.unsqueeze(0), vals.unsqueeze(0)), dim=0).mean(axis=0).unsqueeze(0)
                out = tmp
            offset += size[i]
        return out

def train(model, train_dataset, valid_dataset, batch_size, lr, num_epochs, device, model_file: str, loss_type: str='MSE',  
          shuffle: bool=False, num_workers: int=0, weight_decay: float=5e-4, step_size: int=5, gamma: float=0.1) -> Armadillo:
    """This function execute the training operation

    Args:
        model (_type_): the model to train
        train_dataset (_type_): training dataset generated with train_test_valid_split
        valid_dataset (_type_): validation dataset generated with train_test_valid_split
        batch_size (_type_): size of the batches
        lr (_type_): learning rate
        num_epochs (_type_): number of training epochs
        device (_type_): the device the model is working with
        model_file (str): path to the file where to save the model's checkpoints
        shuffle (bool, optional): if set True train and validation dataset are shuffled. Defaults to False.
        num_workers (int, optional): NA. Defaults to 0.
        weight_decay (float, optional): NA. Defaults to 5e-4.
        step_size (int, optional): number of epochs to wait to update the learning rate. Defaults to 5.
        gamma (float, optional): reduction factor of the learning rate. Defaults to 0.1

    Returns:
        Armadillo: the trained network
    """
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # scheduler
    scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    # loss
    if loss_type == 'MSE':
        print('Using MSELoss criterion')
        loss_criterion = nn.MSELoss()
    elif loss_type == 'MAE':
        print('Using L1Loss criterion')
        loss_criterion = nn.L1Loss()
    else:
        print('Loss criterion not supported')
        raise Exception()

    best_loss = float('inf')
    for epoch in tqdm(range(num_epochs)):
        start_epoch = time.time()
        # Train step
        train_loss = train_epoch(model, train_dataloader, optimizer, loss_criterion, device)

        # Eval step
        
        val_loss = eval_epoch(model, valid_dataloader, loss_criterion, device)

        end_epoch = time.time()

        if val_loss < best_loss:
            best_loss = val_loss
            # save_model_checkpoint
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'hidden_channels' : model.hidden_channels,
                'num_layers' : model.num_layers,
                'dropout' : model.dropout,
                'act' : model.act,
                'gnn_type' : model.gnn_type,
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'in_channels': model.in_channels
            }
            torch.save(checkpoint, model_file)
        scheduler.step()

    return Armadillo(model_file=model_file)

def train_epoch(model: Armadillo, train_dataloader: DataLoader, optimizer: torch.optim.Optimizer, criterion: nn.MSELoss, device: str) -> torch.Tensor:
    """Operations to perform in every training epoch

    Args:
        model (Armadillo): the model to train
        train_dataloader (DataLoader): dataloader containing the training data
        optimizer (torch.optim.Optimizer): optimizer
        criterion (nn.MSELoss): loss criterion
        device (str): the device the model is working with

    Returns:
        torch.Tensor: the average loss of the step
    """
    total_loss = 0
    model.train()
    
    # For each batch of training data...
    for batch in train_dataloader:
        # Forward
        optimizer.zero_grad()
        emb_l = model(batch[0].to(device), batch[3], batch[4], batch[5])
        emb_r = model(batch[1].to(device), batch[6], batch[7], batch[8])

        predictions = F.cosine_similarity(emb_l, emb_r, dim=1)

        predictions[predictions < 0] = 0

        y = batch[2].to(device)

        # Loss
        loss = criterion(predictions, y)
        total_loss += loss.item()

        # Perform a backward pass to calculate gradients
        loss.backward()

        # Update parameters and the learning rate
        optimizer.step()

    # # Calculate the average loss over the entire training data
    avg_train_loss = total_loss / len(train_dataloader)

    return avg_train_loss

def eval_epoch(model: Armadillo, valid_dataloader: DataLoader, criterion: nn.MSELoss, device: str) -> float:
    """Operations to perform evaluation in every epoch

    Args:
        model (Armadillo): the model to evaluate
        valid_dataloader (DataLoader): data loader containing the validation data
        criterion (nn.MSELoss): loss criterion
        device (str): the device the model is working with

    Returns:
        torch.Tensor: the average loss on the validation set 
    """
    avg_loss = 0.0
    model.eval()

    with torch.no_grad():
        for batch in valid_dataloader:
            # to device
            emb_l = model(batch[0].to(device), batch[3], batch[4], batch[5])
            emb_r = model(batch[1].to(device), batch[6], batch[7], batch[8])


            predictions = F.cosine_similarity(emb_l, emb_r, dim=1)

            predictions[predictions < 0] = 0

            y = batch[2].to(device)

            # Loss
            loss = criterion(predictions, y)

            avg_loss += loss.item()

    avg_loss = avg_loss / len(valid_dataloader)

    return avg_loss

def test_bins(model: Armadillo, test_dataset: GraphTriplesDataset, batch_size: int=64, 
         num_workers: int=0, shuffle: bool=False, granularity: float=0.1
         ) -> dict:
    """To compute testing stats in defined overlap intervals

    Args:
        model (Armadillo): the model to test
        test_dataset (GraphTriplesDataset): the full test dataset
        batch_size (int, optional): size of batches. Defaults to 64.
        num_workers (int, optional): NA. Defaults to 0.
        shuffle (bool, optional): to shuffle the dataset. Defaults to False.
        granularity (float, optional): granularity of the bags. Defaults to 0.1.

    Returns:
        dict: dictionary containing the following metrics: mse, mae, variance, min_ae, abs_diff_tensor 
    """
    stats_list = {}
    left = 0
    right = granularity
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    graphs = test_dataset.graphs
    samples = test_dataset.triples
    count = 0
    while right <= 1:
        try:
            tmp = samples[samples['table_overlap'] >= left][:]
            tmp = tmp[tmp['table_overlap'] < right][:]
        except:
            tmp = samples[samples['a%'] >= left][:]
            tmp = tmp[tmp['a%'] < right][:]
        eval_loader = DataLoader(GraphTriplesDataset(tmp, graphs), 
                                batch_size=batch_size,  
                                num_workers=num_workers, 
                                shuffle=shuffle)
        y_pred, y_true = model_inference(
            model,
            eval_loader,
            device
        )
        
        stats_list[str(count)] = compute_metrics(y_pred, y_true)
        
        count += 1
        left += granularity
        right += granularity

    return stats_list

def test(model: Armadillo, test_dataset: GraphTriplesDataset, batch_size: int=64, 
         num_workers: int=0) -> torch.Tensor:
    """Perform the testing operation

    Args:
        model (Armadillo): the model to test
        test_dataset (GraphTriplesDataset): testing dataset generated with train_test_valid_split
        batch_size (int, optional): size of the batches. Defaults to 64.
        num_workers (int, optional): NA. Defaults to 0.

    Returns:
        torch.Tensor: torch.Tensor: the average loss on the test set 
    """
    eval_loader = DataLoader(test_dataset, 
                             batch_size=batch_size,  
                             num_workers=num_workers, 
                             shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model
    # to device

    y_pred, y_true = model_inference(
        model,
        eval_loader,
        device
    )
    return compute_metrics(y_pred, y_true)

def compute_metrics(y_pred: torch.Tensor, y_true: torch.Tensor) -> dict:
    """Function to compute the necessary evaluation metrics

    Args:
        y_pred (torch.Tensor): prediction tensor
        y_true (torch.Tensor): true label tensor

    Returns:
        dict: contains the following metrics: mse, mae, variance, min_ae, abs_diff_tensor 
    """
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mse = F.mse_loss(y_pred, y_true.to(device))
        abs_diff_tensor = torch.abs(y_pred - y_true.to(device))
        mae = torch.mean(abs_diff_tensor)
        variance = torch.var(abs_diff_tensor)
        min_ae = torch.min(abs_diff_tensor)
        max_ae = torch.max(abs_diff_tensor)
    except:
        mse = None
        abs_diff_tensor = None
        mae = None
        variance = None
        min_ae = None
        max_ae = None
    return {'mse':mse, 'mae':mae, 'variance':variance, 'min_ae':min_ae, 'max_ae':max_ae, 'abs_diff_tensor':abs_diff_tensor}

def model_inference(model: Armadillo, data_loader: DataLoader, device: str) -> set:
    """Genertion of 2 sets of labels (predicted and real)

    Args:
        model (Armadillo): the model to test
        data_loader (DataLoader): dataloader containing the test data
        device (str): the device the model is working with

    Returns:
        set: a set of 2 tensors (PREDICTIONS, LABELS)
    """
    model.eval()

    y_pred = None
    y_true = None
    with torch.no_grad():
        for batch in data_loader:
            # to device
            emb_l = model(batch[0].to(device), batch[3], batch[4], batch[5])
            emb_r = model(batch[1].to(device), batch[6], batch[7], batch[8])

            logits = F.cosine_similarity(emb_l, emb_r, dim=1)

            logits[logits < 0] = 0

            y = batch[2]
                # Save the predictions and the labels
            if y_pred is None:
                y_pred = logits  
                y_true = batch[2]
            else:
                y_pred = torch.cat((y_pred, logits))
                y_true = torch.cat((y_true, y))

    return y_pred, y_true

def train_test_pipeline_split(train_file: str, test_file: str, valid_file: str, graph_file: str, model_file: str, hidden_channels: int, num_layers: int,
                        batch_size: int=64, lr: float=0.01, dropout: float=0, initial_embedding_method: str='fasttext',
                        num_epochs: int=100, weight_decay: float=0.0001, act: str='relu',
                        step_size: int=15, gamma: float=0.1, gnn_type: str='GIN', compute_bins_stats: bool=False, relu: bool=False, loss_type: str='MSE') -> Armadillo:
    """This function performs the full train-validate-test pipeline

    Args:
        train_file (str): path to the train triple file
        test_file (str): path to the test triple file
        valid_file (str): path to the validation triple file
        graph_file (str): path to the graph file
        model_file (str): path to the backup file for the model
        test_predictions_file (str): path to the directory containing the logs of the predictions
        hidden_channels (int): size of the generated embeddings
        num_layers (int): number of layers of the network, every embedding will be generated using using his neighbours at distance num_layers
        ttv_ratio (set, optional): a triple that tells the function how to split the dataset (TRAIN, TEST, VALIDATE). Defaults to (0.8,0.1,0.1).
        batch_size (int, optional): number of elements to put in the training batches. Defaults to 64.
        lr (float, optional): learning rate. Defaults to 0.01.
        dropout (float, optional): dropout probability. Defaults to 0.
        num_epochs (int, optional): number of training epochs. Defaults to 100.
        weight_decay (float, optional): NA. Defaults to 0.
        act (str, optional): the activation function used between the layers. Defaults to 'relu'.
        step_size (int, optional): number of epochs to wait to update the learning rate. Defaults to 5.
        gamma (float, optional): reduction factor of the learning rate. Defaults to 0.1
        gnn_type (str): the gnn to use. Defaults to 'GIN'
        compute_bins_stats (bool): set to true to compute stats about intervals of table overlaps. Default to False
        relu (bool, optional): if set to Tre a relu layer will be added at the end of the network, it will prevent negative cosine similarities between the embeddings. Defaults to False.

    Returns:
        Armadillo: the trained network
    """
    set_seed()
    # Load datasets
    print('Loading datasets, it may take some time')
    train_triples = pd.read_csv(train_file)[['r_id','s_id','a%']]
    test_triples = pd.read_csv(test_file)[['r_id','s_id','a%']]
    valid_triples = pd.read_csv(valid_file)[['r_id','s_id','a%']]

    with open(graph_file, 'rb') as f:
        graphs = pickle.load(f)

    train_dataset = GraphTriplesDataset(train_triples, graphs)
    test_dataset = GraphTriplesDataset(test_triples, graphs)
    valid_dataset = GraphTriplesDataset(valid_triples, graphs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = Armadillo(hidden_channels, num_layers, dropout=dropout, act=act, gnn_type=gnn_type, relu=relu, initial_embedding_method=initial_embedding_method)
    start = time.time()

    print('Training starts')

    model = train(model, train_dataset, valid_dataset, batch_size, lr, num_epochs, device, model_file, 
                  weight_decay=weight_decay, step_size=step_size, gamma=gamma, loss_type=loss_type)
    end = time.time()
    t_train=end-start
    print(f'T_train: {t_train}s')

    start = time.time()
    execution_insights_test = test(model, test_dataset, batch_size) 
    mse = execution_insights_test['mse']
    mae = execution_insights_test['mae'] 
    end = time.time()
    t_test = end-start
    print(f'T_test: {t_test}s')
    print(f'MSE: {mse}')
    print(f'MAE: {mae}')

    print('Generating tests for bags')

    execution_insights = {'test':execution_insights_test}
    
    if compute_bins_stats:
        execution_insights_bins = test_bins(model, test_dataset, batch_size)
        execution_insights['bins'] = execution_insights_bins

    execution_insights['test']['model'] = model

    return execution_insights