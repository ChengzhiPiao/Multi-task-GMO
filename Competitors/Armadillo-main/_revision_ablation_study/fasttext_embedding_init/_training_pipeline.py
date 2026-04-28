from _revision_ablation_study.fasttext_embedding_init.armadillo import *

def training_pipeline(train_file: str, test_file: str, valid_file: str, graph_file: str | dict, model_file: str, hidden_channels: int, num_layers: int,
                        batch_size: int=64, lr: float=0.01, dropout: float=0, initial_embedding_method: str='fasttext',
                        num_epochs: int=100, weight_decay: float=0.0001, act: str='relu',
                        step_size: int=15, gamma: float=0.1, gnn_type: str='GIN', compute_bins_stats: bool=False, relu: bool=False, loss_type: str='MAE') -> Armadillo:
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
        log_wandb (bool, optional): if True all the outputs of the experiments will be logged to wandb, an open session is necessary to avoid errors. Defaults to False.
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
    if isinstance(graph_file, str):
        with open(graph_file, 'rb') as f:
            graphs = pickle.load(f)
    else:
        graphs = graph_file

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

def run_Armadillo_experiment_split(train_file: str, test_file: str, valid_file: str, graph_file: str | dict, checkpoint: str, lr: float, batch_size: int,
                         num_epochs: int, out_channels: int, n_layers: int, dropout: float, weight_decay: float, step_size: int, gamma: float, 
                         gnn_type: str, initial_embedding_method: str='sha256', log_wandb=False, relu: bool=False, loss_type: str='MSE') -> None:
    """Utility function to run experiments that will be logged in wandb

    Args:
        project_name (str): name of the project in wandb
        dataset (str): directory containing the stuff necessary to build the dataset
        lr (float): learning rate
        batch_size (int): size of the training batches
        num_epochs (int): number of training epochs
        out_channels (int): size of the embeddings
        n_layers (int): number of layers
        dropout (float): dropout probability
        weight_decay (float): an L2 penalty
        step_size (int): number of epochs to wait to update the learning rate
        gamma (float): reduction factor of the learning rate
        gnn_type (str): the gnn to use, accepted 'GIN' and 'GAT'
        relu (bool, optional): if set to Tre a relu layer will be added at the end of the network, it will prevent negative cosine similarities between the embeddings

    """
    name = checkpoint
    if relu:
        name += "_relu"
    else:
        name += "_no_relu"
    #checkpoint = dataset+f"/{name}.pth"
    print(f'Starting training with {num_epochs} epochs')
    training_pipeline(train_file=train_file, test_file=test_file, valid_file=valid_file, graph_file=graph_file, model_file=checkpoint, hidden_channels=out_channels, num_layers=n_layers, 
                        num_epochs=num_epochs, batch_size=batch_size, lr=lr, dropout=dropout, 
                        weight_decay=weight_decay, step_size=step_size, gamma=gamma, gnn_type=gnn_type, compute_bins_stats=True,
                        relu=relu, initial_embedding_method=initial_embedding_method, loss_type=loss_type)
        