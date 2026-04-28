from Code._training_pipeline import run_Armadillo_experiment_split
from Code._generate_graph_dict import *
import time
import os

def retrain_model(model_out_path: str, train_file: str, test_file: str, valid_file: str, graph_dict: str | dict) -> None:
    """Train a new Armadillo model from scratch

    Args:
        model_out_path (str): path where to save a checpoint containing the weights of the model
        train_file (str): path to the csv file containing the training triples
        test_file (str): path to the csv file containing the testing triples
        valid_file (str): path to the csv file containing the validation triples
        graph_dict (str): path to the dictionary containing the preconstructed table graphs
    """
    print('Model training starting')
    start = time.time()
    loss_type = 'MAE'
    num_epochs = 100
    lr = 0.001
    batch_size = 64
    out_channels = 300
    n_layers = 3
    dropout_prob = 0
    weight_decay = 0.0001
    step_size = 15
    gamma = 0.1
    GNN_type = 'GraphSAGE'
    checkpoint = model_out_path
    initial_embedding_method = 'sha256'
    run_Armadillo_experiment_split(train_file=train_file, test_file=test_file, loss_type=loss_type, valid_file=valid_file, graph_file=graph_dict, 
                                checkpoint=checkpoint, lr=lr, batch_size=batch_size, num_epochs=num_epochs, out_channels=out_channels, n_layers=n_layers, 
                                dropout=dropout_prob, weight_decay=weight_decay, step_size=step_size, gamma=gamma, gnn_type=GNN_type,
                                initial_embedding_method=initial_embedding_method
                                )
    end = time.time()
    print(f'Model trained in {end-start}s')

if __name__ == '__main__':
    """
        Input: root of wikilast or gittables
        Output: a model.pth file containing the weights of new trained model
    """
    root_dataset = ''       # Insert here the full name of the directory containing the train, test, and valid csv files, e.g., root/gittables_root
    table_dict = root_dataset+'/dictionaries/table_dict.pkl'         # Insert here the full name of the table dictionary, e.g., root/gittables_root/dictionaries/table_dict.pkl
    graph_dict = root_dataset+'/dictionaries/table_dict.pkl'         # Insert here the full name of the graph dictionary, e.g., root/gittables_root/dictionaries/graph_dict.pkl
    if not os.path.exists(root_dataset+'/models/'):
        os.makedirs(root_dataset+'/models/')
    model_out = root_dataset+'/models/armadillo.pth'          # Insert here the full name of the model checkpoint, e.g., root/gittables_root/models/armadillo_git.pth
    print('Building graph_dict')
    try:
        with open(graph_dict,'rb') as f:
            graph_dict = pickle.load(f)
    except:
        print('Generating graph dict from scratch')
        graph_dict = generate_graph_dictionary(table_dict_path=table_dict, out_path=graph_dict)
    
    print('Training Starting')
    retrain_model(model_out_path=model_out, train_file=root_dataset+'/train.csv', test_file=root_dataset+'/test.csv', valid_file=root_dataset+'/valid.csv', graph_dict=graph_dict)
    print('Training is complete')
