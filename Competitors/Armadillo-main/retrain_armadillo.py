import os
from Code._csv_preprocessing import generate_table_dict_from_different_folders
from Code._training_pipeline import run_Armadillo_experiment_split
from Code._generate_graph_dict import *
import time

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
    num_epochs = 1
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
    csvs_path = ''  # path to the 'gittables' or 'wikitables' directories, depending on the desired version of armadillo
    table_dictionary_path = ''  # path where to save a table dictionary in .pkl format
    graph_dictionary_path = ''  # path where to save a graph dictionary in .pkl format
    train_test_valid_triples_path = ''   # path containing train.csv, test.csv, and valid.csv, may be the same as csvs_path
    model_out_path = ''  # path where to save the newly trained model

    if csvs_path != '':
        if not os.path.exists(table_dictionary_path):    
            print('Generating table dict from scratch')
            train_dir = csvs_path+'/train/'
            test_dir = csvs_path+'/test'
            valid_dir = csvs_path+'/valid'
            generate_table_dict_from_different_folders([train_dir,test_dir,valid_dir], table_dictionary_path, anon=False)
    try:
        with open(graph_dictionary_path,'rb') as f:
            graph_dict = pickle.load(f)
    except:
        print('Generating graph dict from scratch')
        graph_dict = generate_graph_dictionary(table_dict_path=table_dictionary_path, out_path=graph_dictionary_path)
    
    print('Training Starting')
    retrain_model(model_out_path=model_out_path, train_file=train_test_valid_triples_path+'/train.csv', test_file=train_test_valid_triples_path+'/test.csv', valid_file=train_test_valid_triples_path+'/valid.csv', graph_dict=graph_dict)
    print('Training is complete')