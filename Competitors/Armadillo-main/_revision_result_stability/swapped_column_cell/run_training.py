import sys
sys.path.append(".")
sys.path.append("../../")

from _revision_result_stability.swapped_column_cell._training_pipeline import run_Armadillo_experiment_split
from _revision_result_stability.swapped_column_cell._generate_graph_dict import *
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
    root_ablation = '/home/francesco.pugnaloni/armadillo_all/result_stability/swapped_col_cell/wikitables/' # WikiTables
    # root_ablation = '/home/francesco.pugnaloni/armadillo_all/result_stability/swapped_col_cell/gittables/' # GitTables

    root_dataset = '/home/francesco.pugnaloni/armadillo_all/datasets/WikiTables/'       # WikiTables
    # root_dataset = '/home/francesco.pugnaloni/armadillo_all/datasets/GitTables/' #GitTables

    table_dict = '/home/francesco.pugnaloni/armadillo_all/datasets/WikiTables/dictionaries/table_dict.pkl'         
    # table_dict = '/home/francesco.pugnaloni/armadillo_all/datasets/GitTables//dictionaries/table_dictionaries/table_dict.pkl' #GitTables

    graph_dict = '/home/francesco.pugnaloni/armadillo_all/result_stability/swapped/wikitables/graph_dict.pkl'         # WikiTables
    # graph_dict = '/home/francesco.pugnaloni/armadillo_all/result_stability/swapped/gittables/graph_dict.pkl' #GitTables

    if not os.path.exists(root_ablation+'/models/'):
        os.makedirs(root_ablation+'/models/')
    model_out = root_ablation+'/models/armadillo.pth'          # Insert here the full name of the model checkpoint, e.g., root/gittables_root/models/armadillo_git.pth
    # Create inside the root_dataset directory a log directory if it does not exist
    if not os.path.exists(root_ablation+'/log/'):
        os.makedirs(root_ablation+'/log/')
    print('Building graph_dict')
    try:
        with open(graph_dict,'rb') as f:
            graph_dict = pickle.load(f)
    except:
        print('Generating graph dict from scratch')
        start_time = time.time()
        graph_dict = generate_graph_dictionary(table_dict_path=table_dict, out_path=graph_dict)
        end_time = time.time()
        graph_dict_gen_time = end_time - start_time
        with open(root_ablation+'/log/graph_dict_gen_time.txt','w') as f:
            f.write(str(graph_dict_gen_time))  
        
    print('Training Starting')
    start_time_train = time.time()
    retrain_model(model_out_path=model_out, train_file=root_dataset+'/train.csv', test_file=root_dataset+'/test.csv', valid_file=root_dataset+'/valid.csv', graph_dict=graph_dict)
    end_time_train = time.time()
    train_time = end_time_train - start_time_train
    with open(root_ablation+'/log/train_time.txt','w') as f:
        f.write(str(train_time))
    print('Training is complete')