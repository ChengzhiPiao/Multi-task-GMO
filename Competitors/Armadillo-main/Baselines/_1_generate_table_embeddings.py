import sys
sys.path.append(".")
sys.path.append("../../")
from Baselines.bert.bert_baseline import *
from turl.TURL.turl_table_embedding import TURL_embedding_generator
import pickle
import time

def generate_embedding_dictionary(table_dictionary: str | dict[str, pd.DataFrame], embedding_method: str, outpath: str, gpu_num: str='0', 
                                  out_path_t_execs: str=None) -> dict[str, torch.Tensor]:
    
    if isinstance(table_dictionary, str):
        print(f'Table dictionary: {table_dictionary}')
        with open(table_dictionary, 'rb') as f:
            table_dictionary = pickle.load(f)
    print(f'Embedding method: {embedding_method}')
    print(f'outhpath: {outpath}')
    
    if embedding_method == 'bert_lines':
        model = Bert_table_embedder(max_lines=300, max_columns=300, output_format='mean', granularity='row', output_hidden_states=False, model='bert', gpu_num=gpu_num)
    elif embedding_method == 'bert_cells':
        model = Bert_table_embedder(max_lines=300, max_columns=300, output_format='mean', granularity='cell', output_hidden_states=False, model='bert', gpu_num=gpu_num)
    elif embedding_method == 'bert_tables':
        model = Bert_table_embedder(max_lines=128, max_columns=128, output_format='mean', granularity='table', output_hidden_states=False, model='bert', gpu_num=gpu_num)
    elif embedding_method == 'roberta_lines':
        model = Bert_table_embedder(max_lines=300, max_columns=300, output_format='mean', granularity='row', output_hidden_states=False, model='roberta', gpu_num=gpu_num)
    elif embedding_method == 'roberta_cells':
        model = Bert_table_embedder(max_lines=300, max_columns=300, output_format='mean', granularity='cell', output_hidden_states=False, model='roberta', gpu_num=gpu_num)
    elif embedding_method == 'roberta_tables':
        model = Bert_table_embedder(max_lines=128, max_columns=128, output_format='mean', granularity='table', output_hidden_states=False, model='roberta', gpu_num=gpu_num)
    elif embedding_method == 'turl':
        model = TURL_embedding_generator(sampling_size=-1, data_dir='Baselines/turl/additional_files/turl_datasets', 
                                         model_checkpoint_path='Baselines/turl/additional_files/turl_pretrained.bin',
                                         config_name='Baselines/turl/additional_files/table-base-config_v2.json', 
                                         max_lines=10, gpu_num=gpu_num, max_t_exec=180)
    else:
        raise NotImplementedError
    out = {}
    t_execs = {}
    for k in tqdm(table_dictionary.keys()):
        start = time.time()
        out[k] = model(table_dictionary[k])
        end = time.time()
        t_execs[k] = end-start
    
    with open(outpath, 'wb') as f:
        pickle.dump(out, f)
    if out_path_t_execs != None:
        with open(out_path_t_execs, 'wb') as f:
            pickle.dump(t_execs, f)

    return out

if __name__ == '__main__':
    root_git = '/home/francesco.pugnaloni/armadillo_all/datasets/GitTables/'
    root_wiki = '/home/francesco.pugnaloni/armadillo_all/datasets/WikiTables/'
    #_____________________________ROWS GRANULARITY
    params_bert_rows_300_300_gittables={
        'table_dictionary' : root_git+'/dictionaries/table_dictionaries/table_dict.pkl',
        'embedding_method' : 'bert_lines',
        'outpath' : root_git+'/dictionaries/embedding_dictionaries/emb_dict_bert_lines_300_300.pkl',
        'out_path_t_execs':root_git+'/dictionaries/embedding_dictionaries/t_execs_bert_rows_300_300_gittables.pkl'
    }

    params_roberta_rows_300_300_gittables={
        'table_dictionary' : root_git+'/dictionaries/table_dictionaries/table_dict.pkl',
        'embedding_method' : 'roberta_lines',
        'outpath' : root_git+'/dictionaries/embedding_dictionaries/emb_dict_roberta_lines_300_300.pkl',
        'out_path_t_execs':root_git+'/dictionaries/embedding_dictionaries/t_execs_roberta_rows_300_300_gittables.pkl'
    }

    params_bert_rows_300_300_wikilast={
        'table_dictionary' : root_wiki+'/dictionaries/table_dict.pkl',
        'embedding_method' : 'bert_lines',
        'outpath' : root_wiki+'/dictionaries/embedding_dictionaries/emb_dict_bert_lines_300_300.pkl',
        'out_path_t_execs':root_wiki+'/dictionaries/embedding_dictionaries/t_execs_bert_rows_300_300_wikilast.pkl'
    }

    params_roberta_rows_300_300_wikilast={
        'table_dictionary' : root_wiki+'/dictionaries/table_dict.pkl',
        'embedding_method' : 'roberta_lines',
        'outpath' : root_wiki+'/dictionaries/embedding_dictionaries/emb_dict_roberta_lines_300_300.pkl',
        'out_path_t_execs':root_wiki+'/WikiTables/dictionaries/embedding_dictionaries/t_execs_roberta_rows_300_300_wikilast.pkl'
    }

    #______________________________TABLE GRANULARITY
    params_bert_tables_300_300_gittables={
        'table_dictionary' : root_git+'/dictionaries/table_dictionaries/table_dict.pkl',
        'embedding_method' : 'bert_tables',
        'outpath' : root_git+'/dictionaries/embedding_dictionaries/emb_dict_bert_tables_300_300.pkl',
        'out_path_t_execs':root_git+'/dictionaries/embedding_dictionaries/t_execs_bert_tables_300_300_gittables.pkl'
    }

    params_roberta_tables_300_300_gittables={
        'table_dictionary' : root_git+'/dictionaries/table_dictionaries/table_dict.pkl',
        'embedding_method' : 'roberta_tables',
        'outpath' : root_git+'/dictionaries/embedding_dictionaries/emb_dict_roberta_tables_300_300.pkl',
        'out_path_t_execs':root_git+'/dictionaries/embedding_dictionaries/t_execs_roberta_tables_300_300_gittables.pkl'
    }

    params_bert_tables_300_300_wikilast={
        'table_dictionary' : root_wiki+'/WikiTables/dictionaries/table_dict.pkl',
        'embedding_method' : 'bert_tables',
        'outpath' : root_wiki+'/dictionaries/embedding_dictionaries/emb_dict_bert_tables_300_300.pkl',
        'out_path_t_execs':root_wiki+'/dictionaries/embedding_dictionaries/t_execs_bert_tables_300_300_wikilast.pkl'
    }

    params_roberta_tables_300_300_wikilast={
        'table_dictionary' : root_wiki+'/dictionaries/table_dict.pkl',
        'embedding_method' : 'roberta_tables',
        'outpath' : root_wiki+'/dictionaries/embedding_dictionaries/emb_dict_roberta_tables_300_300.pkl',
        'out_path_t_execs':root_wiki+'/dictionaries/embedding_dictionaries/t_execs_roberta_tables_300_300_wikilast.pkl'
    }

    #_______________________TURL
    params_turl_tables_300_300_gittables={
        'table_dictionary' : root_git+'/dictionaries/table_dictionaries/table_dict.pkl',
        'embedding_method' : 'turl',
        'outpath' : root_git+'/dictionaries/embedding_dictionaries/emb_dict_turl_tables_128_128.pkl',
        'out_path_t_execs':root_git+'/dictionaries/embedding_dictionaries/t_execs_turl_tables_300_300_gittables.pkl'
    }
    params_turl_tables_300_300_wikilast={
        'table_dictionary' : root_wiki+'/dictionaries/table_dict.pkl',
        'embedding_method' : 'turl',
        'outpath' : root_wiki+'/dictionaries/embedding_dictionaries/emb_dict_turl_tables_300_300.pkl',
        'out_path_t_execs':root_wiki+'/dictionaries/embedding_dictionaries/t_execs_turl_tables_300_300_wikilast.pkl'
    }

    #______________________Anonymised methods
    params_roberta_tables_anon_300_300_wikilast={
        'table_dictionary' : root_wiki+'/dictionaries/table_dict_anon.pkl',
        'embedding_method' : 'roberta_tables',
        'outpath' : root_wiki+'/dictionaries/embedding_dictionaries/emb_dict_roberta_tables_anon_300_300.pkl',
        'out_path_t_execs':root_wiki+'/dictionaries/embedding_dictionaries/t_execs_roberta_tables_anon_300_300_wikilast.pkl'
    }
    params_bert_tables_anon_300_300_wikilast={
        'table_dictionary' : '/home/francesco.pugnaloni/armadillo_all/datasets/GitTables/dictionaries/table_dictionaries/missing_tables_turl.pkl',
        'embedding_method' : 'bert_tables',
        'outpath' : root_wiki+'/dictionaries/embedding_dictionaries/emb_dict_bert_tables_anon_300_300.pkl',
        'out_path_t_execs':root_wiki+'/dictionaries/embedding_dictionaries/t_execs_bert_tables_anon_300_300_wikilast.pkl'
    }

    #___________________querying
    params_turl_querying = {
        'table_dictionary' : '/home/francesco.pugnaloni/armadillo_all/datasets/GitTables/table_querying/dictionaries/missing_tables_turl.pkl',
        'embedding_method' : 'turl',
        'outpath' : '/home/francesco.pugnaloni/armadillo_all/datasets/GitTables/table_querying/dictionaries/embedding_dictionaries/missing_embeddings_turl.pkl',
        'out_path_t_execs':'/home/francesco.pugnaloni/armadillo_all/datasets/GitTables/table_querying/dictionaries/embedding_dictionaries/missing_t_execs_turl.pkl'
    }

    # __________________WIKITABLES____________________________

    # generate_embedding_dictionary(**params_bert_rows_300_300_wikilast, gpu_num='0')
    # generate_embedding_dictionary(**params_roberta_rows_300_300_wikilast, gpu_num='0')

    # generate_embedding_dictionary(**params_bert_tables_300_300_wikilast, gpu_num='1')
    # generate_embedding_dictionary(**params_roberta_tables_300_300_wikilast, gpu_num='1')

    # generate_embedding_dictionary(**params_turl_tables_300_300_wikilast, gpu_num='0')

    # generate_embedding_dictionary(**params_roberta_tables_anon_300_300_wikilast, gpu_num='0')
    # generate_embedding_dictionary(**params_bert_tables_anon_300_300_wikilast, gpu_num='0')

    # __________________GITTABLES____________________________
    # generate_embedding_dictionary(**params_bert_tables_300_300_gittables, gpu_num='0')
    # generate_embedding_dictionary(**params_roberta_tables_300_300_gittables, gpu_num='0')

    # generate_embedding_dictionary(**params_bert_rows_300_300_gittables, gpu_num='1')
    # generate_embedding_dictionary(**params_roberta_rows_300_300_gittables, gpu_num='1')

    # generate_embedding_dictionary(**params_turl_tables_300_300_gittables, gpu_num='1')
    
    # _____________________QUERYING

    # generate_embedding_dictionary(**params_turl_querying)

    #_____________________________REVISION EXPERIMENTS_____________________________

    params_bert_tables_anon_300_300_gittables = {
        'table_dictionary' : '/home/francesco.pugnaloni/armadillo_all/datasets/GitTables/dictionaries/table_dictionaries/table_dict_anon.pkl',
        'embedding_method' : 'bert_tables',
        'outpath' : '/home/francesco.pugnaloni/armadillo_all/datasets/GitTables/dictionaries/embedding_dictionaries/emb_dict_bert_tables_anon_300_300.pkl',
        'out_path_t_execs':'/home/francesco.pugnaloni/armadillo_all/datasets/GitTables/dictionaries/embedding_dictionaries/t_execs_bert_tables_anon_300_300.pkl'
    }

    params_roberta_tables_anon_300_300_gittables = {
        'table_dictionary' : '/home/francesco.pugnaloni/armadillo_all/datasets/GitTables/dictionaries/table_dictionaries/table_dict_anon.pkl',
        'embedding_method' : 'roberta_tables',
        'outpath' : '/home/francesco.pugnaloni/armadillo_all/datasets/GitTables/dictionaries/embedding_dictionaries/emb_dict_roberta_tables_anon_300_300.pkl',
        'out_path_t_execs':'/home/francesco.pugnaloni/armadillo_all/datasets/GitTables/dictionaries/embedding_dictionaries/t_execs_roberta_tables_anon_300_300.pkl'
    }

    params_bert_tables_anon_300_300_querying = {
        'table_dictionary' : '/home/francesco.pugnaloni/armadillo_all/datasets/GitTables/table_querying/dictionaries/table_dict_table_querying_anon.pkl',
        'embedding_method' : 'bert_tables',
        'outpath' : '/home/francesco.pugnaloni/armadillo_all/datasets/GitTables/table_querying/dictionaries/embedding_dictionaries/emb_dict_bert_tables_anon_300_300.pkl',
        'out_path_t_execs':'/home/francesco.pugnaloni/armadillo_all/datasets/GitTables/table_querying/dictionaries/embedding_dictionaries/t_execs_bert_tables_anon_300_300.pkl.pkl'
    }

    params_roberta_tables_anon_300_300_querying = {
        'table_dictionary' : '/home/francesco.pugnaloni/armadillo_all/datasets/GitTables/table_querying/dictionaries/table_dict_table_querying_anon.pkl',
        'embedding_method' : 'roberta_tables',
        'outpath' : '/home/francesco.pugnaloni/armadillo_all/datasets/GitTables/table_querying/dictionaries/embedding_dictionaries/emb_dict_roberta_tables_anon_300_300.pkl',
        'out_path_t_execs':'/home/francesco.pugnaloni/armadillo_all/datasets/GitTables/table_querying/dictionaries/embedding_dictionaries/t_execs_roberta_tables_anon_300_300.pkl.pkl'
    }

    # print('Now embedding params_bert_tables_anon_300_300_gittables')
    # generate_embedding_dictionary(**params_bert_tables_anon_300_300_gittables, gpu_num='1')
    # print('Now embedding params_bert_tables_anon_300_300_querying')
    # generate_embedding_dictionary(**params_bert_tables_anon_300_300_querying, gpu_num='1')

    # print('Now embedding params_roberta_tables_anon_300_300_gittables')
    # generate_embedding_dictionary(**params_roberta_tables_anon_300_300_gittables, gpu_num='1')
    # print('Now embedding params_roberta_tables_anon_300_300_querying')
    # generate_embedding_dictionary(**params_roberta_tables_anon_300_300_querying, gpu_num='1')

    # ________________________________________________________________________________________________________________________________




    model = TURL_embedding_generator(sampling_size=-1, data_dir='Baselines/turl/additional_files/turl_datasets', 
                                         model_checkpoint_path='Baselines/turl/additional_files/turl_pretrained.bin',
                                         config_name='Baselines/turl/additional_files/table-base-config_v2.json', gpu_num='1')
    # test = pd.read_csv('/home/francesco.pugnaloni/armadillo_all/datasets/gittables/csv/train_csv/abstraction_csv_licensed.zip_01-02_67.csv')

    # kk = model(test)
    
    # print('End')
