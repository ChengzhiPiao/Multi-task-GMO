import pickle
from Baselines._4_prepare_t_exec_file import *
from Baselines._1_generate_table_embeddings import *
from Baselines.turl.TURL.turl_table_embedding import *

def prepare_embeddings_table_querying(table_dict_table_querying, embedding_dict_old, t_execs_dict_old, embedding_method, out_emb_path, out_t_execs_path, gpu_num: str='0'):
    if embedding_method == 'bert_lines':
        model = Bert_table_embedder(max_lines=300, max_columns=300, output_format='mean', granularity='row', output_hidden_states=False, model='bert', gpu_num=gpu_num)
    elif embedding_method == 'bert_cells':
        model = Bert_table_embedder(max_lines=300, max_columns=300, output_format='mean', granularity='cell', output_hidden_states=False, model='bert', gpu_num=gpu_num)
    elif embedding_method == 'bert_tables':
        model = Bert_table_embedder(max_lines=300, max_columns=300, output_format='mean', granularity='table', output_hidden_states=False, model='bert', gpu_num=gpu_num)
    elif embedding_method == 'roberta_lines':
        model = Bert_table_embedder(max_lines=300, max_columns=300, output_format='mean', granularity='row', output_hidden_states=False, model='roberta', gpu_num=gpu_num)
    elif embedding_method == 'roberta_cells':
        model = Bert_table_embedder(max_lines=300, max_columns=300, output_format='mean', granularity='cell', output_hidden_states=False, model='roberta', gpu_num=gpu_num)
    elif embedding_method == 'roberta_tables':
        model = Bert_table_embedder(max_lines=300, max_columns=300, output_format='mean', granularity='table', output_hidden_states=False, model='roberta', gpu_num=gpu_num)
    elif embedding_method == 'turl':
        model = TURL_embedding_generator(sampling_size=-1, data_dir='Baselines/turl/additional_files/turl_datasets', 
                                         model_checkpoint_path='Baselines/turl/additional_files/turl_pretrained.bin',
                                         config_name='Baselines/turl/additional_files/table-base-config_v2.json', 
                                         max_lines=300, gpu_num=gpu_num, max_t_exec=180)
    else:
        raise NotImplementedError    

    if isinstance(table_dict_table_querying,str):
        with open(table_dict_table_querying, 'rb') as f:
            table_dict_table_querying = pickle.load(f)

    if isinstance(embedding_dict_old,str):
        with open(embedding_dict_old, 'rb') as f:
            embedding_dict_old = pickle.load(f)

    if isinstance(t_execs_dict_old,str):
        with open(t_execs_dict_old, 'rb') as f:
            t_execs_dict_old = pickle.load(f)

    out_emb = {}
    out_t_execs = {}
    for k in tqdm(table_dict_table_querying.keys()):
        try:
            out_emb[k] = embedding_dict_old[k]
            out_t_execs[k] = t_execs_dict_old[k]
        except:
            start = time.time()
            out_emb[k] = model(table_dict_table_querying[k])
            end = time.time()
            out_t_execs[k] = end-start
    with open(out_emb_path, 'wb') as f:
        pickle.dump(out_emb, f)
    with open(out_t_execs_path, 'wb') as f:
        pickle.dump(out_t_execs, f)

turl = {
    'table_dict_table_querying':'/home/francesco.pugnaloni/armadillo_all/datasets/GitTables/table_querying/dictionaries/table_dict_table_querying.pkl',
    'embedding_dict_old':'/home/francesco.pugnaloni/armadillo_all/datasets/GitTables/dictionaries/embedding_dictionaries/emb_dict_turl_tables_128_128.pkl',
    't_execs_dict_old':'/home/francesco.pugnaloni/armadillo_all/datasets/GitTables/dictionaries/embedding_dictionaries/t_execs_turl_tables_300_300_gittables.pkl',
    'embedding_method':'turl',
    'out_emb_path':'/home/francesco.pugnaloni/armadillo_all/datasets/GitTables/table_querying/dictionaries/embedding_dictionaries/emb_dict_turl.pkl',
    'out_t_execs_path':'/home/francesco.pugnaloni/armadillo_all/datasets/GitTables/table_querying/dictionaries/embedding_dictionaries/t_execs_turl.pkl'
}
bert_r={
    'table_dict_table_querying':'/home/francesco.pugnaloni/armadillo_all/datasets/GitTables/table_querying/dictionaries/table_dict_table_querying.pkl',
    'embedding_dict_old':'/home/francesco.pugnaloni/armadillo_all/datasets/GitTables/dictionaries/embedding_dictionaries/emb_dict_bert_lines_300_300.pkl',
    't_execs_dict_old':'/home/francesco.pugnaloni/armadillo_all/datasets/GitTables/dictionaries/embedding_dictionaries/t_execs_bert_rows_300_300_gittables.pkl',
    'embedding_method':'bert_lines',
    'out_emb_path':'/home/francesco.pugnaloni/armadillo_all/datasets/GitTables/table_querying/dictionaries/embedding_dictionaries/emb_dict_bert_r.pkl',
    'out_t_execs_path':'/home/francesco.pugnaloni/armadillo_all/datasets/GitTables/table_querying/dictionaries/embedding_dictionaries/t_execs_bert_r.pkl'
}
bert_t = {
    'table_dict_table_querying':'/home/francesco.pugnaloni/armadillo_all/datasets/GitTables/table_querying/dictionaries/table_dict_table_querying.pkl',
    'embedding_dict_old':'/home/francesco.pugnaloni/armadillo_all/datasets/GitTables/dictionaries/embedding_dictionaries/emb_dict_bert_tables_300_300.pkl',
    't_execs_dict_old':'/home/francesco.pugnaloni/armadillo_all/datasets/GitTables/dictionaries/embedding_dictionaries/t_execs_bert_tables_300_300_gittables.pkl',
    'embedding_method':'bert_tables',
    'out_emb_path':'/home/francesco.pugnaloni/armadillo_all/datasets/GitTables/table_querying/dictionaries/embedding_dictionaries/emb_dict_bert_t.pkl',
    'out_t_execs_path':'/home/francesco.pugnaloni/armadillo_all/datasets/GitTables/table_querying/dictionaries/embedding_dictionaries/t_execs_bert_t.pkl'
}
roberta_r={
    'table_dict_table_querying':'/home/francesco.pugnaloni/armadillo_all/datasets/GitTables/table_querying/dictionaries/table_dict_table_querying.pkl',
    'embedding_dict_old':'/home/francesco.pugnaloni/armadillo_all/datasets/GitTables/dictionaries/embedding_dictionaries/emb_dict_roberta_lines_300_300.pkl',
    't_execs_dict_old':'/home/francesco.pugnaloni/armadillo_all/datasets/GitTables/dictionaries/embedding_dictionaries/t_execs_roberta_rows_300_300_gittables.pkl',
    'embedding_method':'roberta_lines',
    'out_emb_path':'/home/francesco.pugnaloni/armadillo_all/datasets/GitTables/table_querying/dictionaries/embedding_dictionaries/emb_dict_roberta_r.pkl',
    'out_t_execs_path':'/home/francesco.pugnaloni/armadillo_all/datasets/GitTables/table_querying/dictionaries/embedding_dictionaries/t_execs_roberta_r.pkl'
}
roberta_t={
    'table_dict_table_querying':'/home/francesco.pugnaloni/armadillo_all/datasets/GitTables/table_querying/dictionaries/table_dict_table_querying.pkl',
    'embedding_dict_old':'/home/francesco.pugnaloni/armadillo_all/datasets/GitTables/dictionaries/embedding_dictionaries/emb_dict_roberta_tables_300_300.pkl',
    't_execs_dict_old':'/home/francesco.pugnaloni/armadillo_all/datasets/GitTables/dictionaries/embedding_dictionaries/t_execs_roberta_tables_300_300_gittables.pkl',
    'embedding_method':'roberta_tables',
    'out_emb_path':'/home/francesco.pugnaloni/armadillo_all/datasets/GitTables/table_querying/dictionaries/embedding_dictionaries/emb_dict_roberta_t.pkl',
    'out_t_execs_path':'/home/francesco.pugnaloni/armadillo_all/datasets/GitTables/table_querying/dictionaries/embedding_dictionaries/t_execs_roberta_t.pkl'
}
model_list = [
    turl,
    # bert_r,
    # bert_t,
    # roberta_r,
    # roberta_t
]
for m in model_list:
    print(f'Working on model {m["embedding_method"]}')
    prepare_embeddings_table_querying(**m)
