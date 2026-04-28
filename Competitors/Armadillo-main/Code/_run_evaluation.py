from _generate_graph_dict import generate_graph_dictionary
from _embed_all_no_paral import run_experiment
from _training_pipeline import run_Armadillo_experiment_split
from _overlap_of_a_pair_effe_effi_processing import recompute_embeddings_overlaps_overlap_computation_time, repeat_test_emb_already_computed
from _performance_overlap_computation import predict_overlap_compute_AE
from _table_stats_computation import compute_tables_stats
import pandas as pd
import pickle
import time

def run_all(operations: list=['build_graph_dict', 'build_embedding_dict', 'efficiency_tests', 'effectiveness_tests'], table_dict: str | dict=None, graph_dict: str=None, embedding_generation_method: str='sha256', model_file: str=None,
            train_file: str=None, test_file: str=None, valid_file: str=None, num_epochs: int=50, embedding_file: str=None, plot_data_emb_gen: str=None, sloth_output_file: str=None,
            new_model_trained: str= None,
            plot_data_efficiency: str=None, loss_type: str='MAE', triple_dataset_without_predictions: str=None, table_stats_file: str=None, plot_data_effectiveness: str=None, dropout: float=0, gnn_type: str='GraphSAGE'):

    if 'build_graph_dict' in operations:
        print('Starting graph construction')
        start = time.time()
        generate_graph_dictionary(table_dict_path=table_dict, out_path=graph_dict, embedding_generation_method=embedding_generation_method)
        end = time.time()
        print(f'Graph built in {end-start}s')

    if 'train_model' in operations:
        print('Model training starting')
        start = time.time()
        name = 'Armadillo'
        lr = 0.001
        batch_size = 64
        out_channels = 300
        n_layers = 3
        dropout_prob = dropout
        weight_decay = 0.0001
        step_size = 15
        gamma = 0.1
        GNN_type = gnn_type
        checkpoint = new_model_trained
        log_wandb = True
        initial_embedding_method = embedding_generation_method
        run_Armadillo_experiment_split(project_name=name, train_file=train_file, test_file=test_file, loss_type=loss_type, valid_file=valid_file, graph_file=graph_dict, 
                                    checkpoint=checkpoint, lr=lr, batch_size=batch_size, num_epochs=num_epochs, out_channels=out_channels, n_layers=n_layers, 
                                    dropout=dropout_prob, weight_decay=weight_decay, step_size=step_size, gamma=gamma, gnn_type=GNN_type,
                                    log_wandb=log_wandb, initial_embedding_method=initial_embedding_method
                                    )
        end = time.time()
        print(f'Model trained in {end-start}s')
    
    if 'build_embedding_dict' in operations:
        start = time.time()
        print('Embedding dictionary construction starting')
        run_experiment(model_file=model_file, table_dict_path=table_dict, embedding_file=embedding_file, experiment_data_file_path=plot_data_emb_gen, graphs_path=graph_dict)
        end = time.time()
        print(f'Model trained in {end-start}s')

    if 'efficiency_tests' in operations:
        print('Efficiency Tests starting, do not run other processes')
        recompute_embeddings_overlaps_overlap_computation_time(
            sloth_outputs_file=sloth_output_file,
            model_file=model_file,
            table_dict=table_dict,
            output_file=plot_data_efficiency
        )
        repeat_test_emb_already_computed(old_file=plot_data_efficiency,
                                     embeddings_dict=embedding_file,
                                     out_path=plot_data_efficiency)
        
    if 'compute_tables_stats' in operations:
        print('Table stats computation starting')
        compute_tables_stats(table_dict=table_dict, outpath=table_stats_file)

    if 'effectiveness_tests' in operations:
        print('Effectiveness Tests starting')
        tdnp = pd.read_csv(triple_dataset_without_predictions)
        if tdnp.shape[1] == 4:
            tdnp = tdnp.drop(tdnp.columns[0], axis=1)
            tdnp = tdnp.fillna(0)
        predict_overlap_compute_AE(unlabelled=tdnp, embedding_dict=embedding_file, out_path=plot_data_effectiveness)
    
    if 'generate_charts' in operations:
        print('Charts generation starting')
        
    
    print('All operation executed')


if __name__ == '__main__':
    td = {
        'a':pd.DataFrame({'a':[1,2,3,4], 'b':['gatto','cane', 'iena','pollo']}),
        'b':pd.DataFrame({'nutria':[7,8,9], 'armadillo':[1,2,3]})
    }
    with open('/home/francesco.pugnaloni/tmp/tables_test.pkl', 'wb') as f:
        pickle.dump(td,f)
    pd.DataFrame({'r_id':['a'], 's_id':['b'], 'a%':[0.3]}).to_csv('/home/francesco.pugnaloni/tmp/train.csv', index=False)
    pd.DataFrame({'r_id':['a'], 's_id':['b'], 'a%':[0.3]}).to_csv('/home/francesco.pugnaloni/tmp/test.csv', index=False)
    pd.DataFrame({'r_id':['a'], 's_id':['b'], 'a%':[0.3]}).to_csv('/home/francesco.pugnaloni/tmp/valid.csv', index=False)

    root = '/home/francesco.pugnaloni/tmp/'

    run_all(
        operations=[
            'build_table_dict',
            'build_graph_dict',
            'train_model',
            'build_embedding_dict',
            'efficiency_tests',
            'compute_tables_stats',
            'effectiveness_tests'
        ],
        loss_type='MAE',
        new_model_trained=root+'/model.pth',
        table_dict=root+'/tables_test.pkl',
        graph_dict=root+'/graph_dict.pkl',
        embedding_generation_method='sha256',
        model_file=root+'/model.pth',
        train_file=root+'/train.csv',
        test_file=root+'/test.csv',
        valid_file=root+'/valid.csv',
        embedding_file=root+'/embeddings.pkl',
        plot_data_emb_gen=root+'/emb_gen_plot_data.csv',
        plot_data_effectiveness=root+'/effectiveness.csv',
        plot_data_efficiency=root+'/efficiency.csv',
        sloth_output_file=root+'/test.csv',
        triple_dataset_without_predictions=root+'/test.csv',
        table_stats_file=root+'/stats.csv',
        num_epochs=100
    )