from Baselines._1_generate_table_embeddings import generate_embedding_dictionary
from Baselines._3_evaluate_scaling_models import only_MAE
import pandas as pd
import pickle

def test_only_dict(table_dict, test_triples):
    test_triples = pd.read_csv(test_triples)
    test_set = set(test_triples['r_id']).union(set(test_triples['s_id']))
    out = {k:table_dict[k] for k in test_set}
    return out

def cross_domain_eval(table_dictionary: dict|str,embedding_method:str,embedding_path:str,out_path_t_execs:str, model_name:str,model_checkpoint:str, test_dataset_path:str, gpu_num: str='1') -> dict:
    try:
        with open(embedding_path, 'rb') as f:
            embedding_path = pickle.load(f)
    except:
        if isinstance(table_dictionary, str):
            with open(table_dictionary,'rb') as f:
                table_dictionary = pickle.load(f)
        table_dictionary = test_only_dict(table_dictionary, test_dataset_path)
        generate_embedding_dictionary(table_dictionary=table_dictionary, embedding_method=embedding_method, outpath=embedding_path, out_path_t_execs=out_path_t_execs, gpu_num=gpu_num)
    
    MAE = only_MAE(model_name=model_name, model_checkpoint=model_checkpoint, embedding_path=embedding_path,test_dataset_path=test_dataset_path)
    print(MAE)

if __name__ == '__main__':
    root_git = '/home/francesco.pugnaloni/armadillo_all/datasets/GitTables/'
    models_git= '/home/francesco.pugnaloni/armadillo_all/datasets/GitTables/models/'
    embeddings_git = root_git+'dictionaries/embedding_dictionaries/'
    test_git = root_git+'test.csv'

    root_wiki = '/home/francesco.pugnaloni/armadillo_all/datasets/WikiTables/'
    models_wiki = '/home/francesco.pugnaloni/armadillo_all/datasets/WikiTables/models/'
    embeddings_wiki = root_wiki+'dictionaries/embedding_dictionaries/'
    test_wiki = root_wiki+'test.csv'

    bert_tg_on_wiki =  {
        'table_dictionary':root_wiki+'/dictionaries/table_dict.pkl',
        'embedding_method':'bert_tables',
        'embedding_path':root_wiki+'/dictionaries/embedding_dictionaries/bert_tg_on_wiki.pkl',
        'out_path_t_execs':root_wiki+'/dictionaries/embedding_dictionaries/t_execs_bert_tg_on_wiki.pkl',
        'model_name':'bert_tables_300_300_gittables',
        'model_checkpoint':root_git+'/models/bert_tables_300_300_gittables.pth',
        'test_dataset_path':root_wiki+'/test.csv'
    }
    bert_tw_on_git = {
        'table_dictionary':root_git+'/dictionaries/table_dictionaries/table_dict.pkl',
        'embedding_method':'bert_tables',
        'embedding_path':root_git+'/dictionaries/embedding_dictionaries/bert_tw_on_git.pkl',
        'out_path_t_execs':root_git+'/dictionaries/embedding_dictionaries/t_execs_bert_tw_on_git.pkl',
        'model_name':'bert_tables_300_300_wikitables',
        'model_checkpoint':root_wiki+'/models/bert_tables_300_300_wikitables.pth',
        'test_dataset_path':root_git+'/test.csv'
    }
    roberta_tg_on_wiki = {
        'table_dictionary':root_wiki+'/dictionaries/table_dict.pkl',
        'embedding_method':'roberta_tables',
        'embedding_path':root_wiki+'/dictionaries/embedding_dictionaries/roberta_tg_on_wiki.pkl',
        'out_path_t_execs':root_wiki+'/dictionaries/embedding_dictionaries/t_execs_roberta_tg_on_wiki.pkl',
        'model_name':'roberta_tables_300_300_gittables',
        'model_checkpoint':root_git+'/models/roberta_tables_300_300_gittables.pth',
        'test_dataset_path':root_wiki+'/test.csv'
    }
    roberta_tw_on_git = {
        'table_dictionary':root_git+'/dictionaries/table_dictionaries/table_dict.pkl',
        'embedding_method':'roberta_tables',
        'embedding_path':root_git+'/dictionaries/embedding_dictionaries/roberta_tw_on_git.pkl',
        'out_path_t_execs':root_git+'/dictionaries/embedding_dictionaries/t_execs_roberta_tw_on_git.pkl',
        'model_name':'roberta_tables_300_300_wikitables',
        'model_checkpoint':root_wiki+'/models/roberta_tables_300_300_wikitables.pth',
        'test_dataset_path':root_git+'/test.csv'
    }
    cross_domain_eval(**bert_tg_on_wiki)