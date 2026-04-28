import sys
sys.path.append(".")
sys.path.append("../../")
from Baselines.embdi.adapter_embdi import generate_table_embeddings
import pandas as pd
from tqdm import tqdm
import pickle
import time
SCENARIO='/home/francesco.pugnaloni/armadillo_all/Armadillo_local/Baselines/embdi/config-dblp_acm-sm'

def embed_triple_file(triple_file: pd.DataFrame|str, table_dict: dict|str,cache_directory_mount: str) -> set:
    if isinstance(triple_file, str):
        triple_file = pd.read_csv(triple_file)
    if isinstance(table_dict, str):
        with open(table_dict, 'rb') as f:
            table_dict = pickle.load(f)
    embedding_dict = {}
    t_execs = {}
    for r in tqdm(range(triple_file.shape[0])):
        sample = triple_file.iloc[r]
        r_id = sample.loc['r_id']
        s_id = sample.loc['s_id']
        k = f'{r_id}|{s_id}'
        t_1 = table_dict[r_id]
        t_2 = table_dict[s_id]
        start = time.time()
        e_1, e_2 = generate_table_embeddings(SCENARIO, df_1=t_1, df_2=t_2,cache_directory_mount=cache_directory_mount)
        end = time.time()
        t_execs[k] = end-start
        embedding_dict[k] = (e_1,e_2)
        print('___________________________________________________________________________________________________________________________________________________________________________')
        print('___________________________________________________________________________________________________________________________________________________________________________')
    return embedding_dict, t_execs

def embed_triple_files_list(triple_files_list: list, table_dict: dict|str, out_path_embedding_dict: str=None, out_path_t_exec: str=None, cache_directory_mount: str=None) -> set:
    if isinstance(table_dict, str):
        with open(table_dict, 'rb') as f:
            table_dict = pickle.load(f)
    out = {}
    t_execs = {}
    for f in triple_files_list:
        print('_________________________________________________________')
        print(f'Processing {f}')
        print('_________________________________________________________')
        new_embs, new_times = embed_triple_file(f,table_dict,cache_directory_mount)
        out.update(new_embs)
        t_execs.update(new_times)

    if out_path_embedding_dict != None:
        with open(out_path_embedding_dict, 'wb') as f:
            pickle.dump(out, f)

    if out_path_t_exec != None:
        with open(out_path_t_exec,'wb') as f:
            pickle.dump(t_execs, f)

    return out, t_execs

if __name__ == '__main__':

    blocks = [(0,50_000),(50_000,100_000),(100_000,150_000),(150_000,200_000),(200_000,250_000),(250_000,300_000),(300_000,350_000),(350_000,400_000),(400_000,450_000),(450_000,500_000)]
    train = pd.read_csv('/home/francesco.pugnaloni/armadillo_all/datasets/WikiTables/train.csv')
    # __test__ = train.iloc[0:9]
    # name = '_mockup'
    # params_mockup = {
    #     'triple_files_list':[__test__],
    #     'table_dict':'/home/francesco.pugnaloni/armadillo_all/datasets/WikiTables/dictionaries/table_dict.pkl',
    #     'cache_directory_mount':'/home/francesco.pugnaloni/armadillo_all/tmp/embdi_caches/cache'+name+'/',
    #     'out_path_embedding_dict':'/home/francesco.pugnaloni/armadillo_all/datasets/WikiTables/dictionaries/embedding_dictionaries/embdi/embedding_dict'+name+'.pkl',
    #     'out_path_t_exec':'/home/francesco.pugnaloni/armadillo_all/datasets/WikiTables/dictionaries/embedding_dictionaries/embdi/t_execs'+name+'.pkl'
    # }
    t_0 = train.iloc[blocks[0][0]:blocks[0][1]]
    name = '_train_0'
    params_t_0 = {
        'triple_files_list':[t_0],
        'table_dict':'/home/francesco.pugnaloni/armadillo_all/datasets/WikiTables/dictionaries/table_dict.pkl',
        'cache_directory_mount':'/home/francesco.pugnaloni/armadillo_all/tmp/embdi_caches/cache'+name+'/',
        'out_path_embedding_dict':'/home/francesco.pugnaloni/armadillo_all/datasets/WikiTables/dictionaries/embedding_dictionaries/embdi/embedding_dict'+name+'.pkl',
        'out_path_t_exec':'/home/francesco.pugnaloni/armadillo_all/datasets/WikiTables/dictionaries/embedding_dictionaries/embdi/t_execs'+name+'.pkl'
    }

    t_1 = train.iloc[blocks[1][0]:blocks[1][1]]
    name = '_train_1'
    params_t_1 = {
        'triple_files_list':[t_1],
        'table_dict':'/home/francesco.pugnaloni/armadillo_all/datasets/WikiTables/dictionaries/table_dict.pkl',
        'cache_directory_mount':'/home/francesco.pugnaloni/armadillo_all/tmp/embdi_caches/cache'+name+'/',
        'out_path_embedding_dict':'/home/francesco.pugnaloni/armadillo_all/datasets/WikiTables/dictionaries/embedding_dictionaries/embdi/embedding_dict'+name+'.pkl',
        'out_path_t_exec':'/home/francesco.pugnaloni/armadillo_all/datasets/WikiTables/dictionaries/embedding_dictionaries/embdi/t_execs'+name+'.pkl'
    }

    t_2 = train.iloc[blocks[2][0]:blocks[2][1]]
    name = '_train_2'
    params_t_2 = {
        'triple_files_list':[t_2],
        'table_dict':'/home/francesco.pugnaloni/armadillo_all/datasets/WikiTables/dictionaries/table_dict.pkl',
        'cache_directory_mount':'/home/francesco.pugnaloni/armadillo_all/tmp/embdi_caches/cache'+name+'/',
        'out_path_embedding_dict':'/home/francesco.pugnaloni/armadillo_all/datasets/WikiTables/dictionaries/embedding_dictionaries/embdi/embedding_dict'+name+'.pkl',
        'out_path_t_exec':'/home/francesco.pugnaloni/armadillo_all/datasets/WikiTables/dictionaries/embedding_dictionaries/embdi/t_execs'+name+'.pkl'
    }

    t_3 = train.iloc[blocks[3][0]:blocks[3][1]]
    name = '_train_3'
    params_t_3 = {
        'triple_files_list':[t_3],
        'table_dict':'/home/francesco.pugnaloni/armadillo_all/datasets/WikiTables/dictionaries/table_dict.pkl',
        'cache_directory_mount':'/home/francesco.pugnaloni/armadillo_all/tmp/embdi_caches/cache'+name+'/',
        'out_path_embedding_dict':'/home/francesco.pugnaloni/armadillo_all/datasets/WikiTables/dictionaries/embedding_dictionaries/embdi/embedding_dict'+name+'.pkl',
        'out_path_t_exec':'/home/francesco.pugnaloni/armadillo_all/datasets/WikiTables/dictionaries/embedding_dictionaries/embdi/t_execs'+name+'.pkl'
    }

    t_4 = train.iloc[blocks[4][0]:blocks[4][1]]
    name = '_train_4'
    params_t_4 = {
        'triple_files_list':[t_4],
        'table_dict':'/home/francesco.pugnaloni/armadillo_all/datasets/WikiTables/dictionaries/table_dict.pkl',
        'cache_directory_mount':'/home/francesco.pugnaloni/armadillo_all/tmp/embdi_caches/cache'+name+'/',
        'out_path_embedding_dict':'/home/francesco.pugnaloni/armadillo_all/datasets/WikiTables/dictionaries/embedding_dictionaries/embdi/embedding_dict'+name+'.pkl',
        'out_path_t_exec':'/home/francesco.pugnaloni/armadillo_all/datasets/WikiTables/dictionaries/embedding_dictionaries/embdi/t_execs'+name+'.pkl'
    }

    t_5 = train.iloc[blocks[5][0]:blocks[5][1]]
    name = '_train_5'
    params_t_5 = {
        'triple_files_list':[t_5],
        'table_dict':'/home/francesco.pugnaloni/armadillo_all/datasets/WikiTables/dictionaries/table_dict.pkl',
        'cache_directory_mount':'/home/francesco.pugnaloni/armadillo_all/tmp/embdi_caches/cache'+name+'/',
        'out_path_embedding_dict':'/home/francesco.pugnaloni/armadillo_all/datasets/WikiTables/dictionaries/embedding_dictionaries/embdi/embedding_dict'+name+'.pkl',
        'out_path_t_exec':'/home/francesco.pugnaloni/armadillo_all/datasets/WikiTables/dictionaries/embedding_dictionaries/embdi/t_execs'+name+'.pkl'
    }

    t_6 = train.iloc[blocks[6][0]:blocks[6][1]]
    name = '_train_6'
    params_t_6 = {
        'triple_files_list':[t_6],
        'table_dict':'/home/francesco.pugnaloni/armadillo_all/datasets/WikiTables/dictionaries/table_dict.pkl',
        'cache_directory_mount':'/home/francesco.pugnaloni/armadillo_all/tmp/embdi_caches/cache'+name+'/',
        'out_path_embedding_dict':'/home/francesco.pugnaloni/armadillo_all/datasets/WikiTables/dictionaries/embedding_dictionaries/embdi/embedding_dict'+name+'.pkl',
        'out_path_t_exec':'/home/francesco.pugnaloni/armadillo_all/datasets/WikiTables/dictionaries/embedding_dictionaries/embdi/t_execs'+name+'.pkl'
    }

    t_7 = train.iloc[blocks[7][0]:blocks[7][1]]
    name = '_train_7'
    params_t_7 = {
        'triple_files_list':[t_7],
        'table_dict':'/home/francesco.pugnaloni/armadillo_all/datasets/WikiTables/dictionaries/table_dict.pkl',
        'cache_directory_mount':'/home/francesco.pugnaloni/armadillo_all/tmp/embdi_caches/cache'+name+'/',
        'out_path_embedding_dict':'/home/francesco.pugnaloni/armadillo_all/datasets/WikiTables/dictionaries/embedding_dictionaries/embdi/embedding_dict'+name+'.pkl',
        'out_path_t_exec':'/home/francesco.pugnaloni/armadillo_all/datasets/WikiTables/dictionaries/embedding_dictionaries/embdi/t_execs'+name+'.pkl'
    }

    t_8 = train.iloc[blocks[8][0]:blocks[8][1]]
    name = '_train_8'
    params_t_8 = {
        'triple_files_list':[t_8],
        'table_dict':'/home/francesco.pugnaloni/armadillo_all/datasets/WikiTables/dictionaries/table_dict.pkl',
        'cache_directory_mount':'/home/francesco.pugnaloni/armadillo_all/tmp/embdi_caches/cache'+name+'/',
        'out_path_embedding_dict':'/home/francesco.pugnaloni/armadillo_all/datasets/WikiTables/dictionaries/embedding_dictionaries/embdi/embedding_dict'+name+'.pkl',
        'out_path_t_exec':'/home/francesco.pugnaloni/armadillo_all/datasets/WikiTables/dictionaries/embedding_dictionaries/embdi/t_execs'+name+'.pkl'
    }

    t_9 = train.iloc[blocks[9][0]:blocks[9][1]]
    name = '_train_9'
    params_t_9 = {
        'triple_files_list':[t_9],
        'table_dict':'/home/francesco.pugnaloni/armadillo_all/datasets/WikiTables/dictionaries/table_dict.pkl',
        'cache_directory_mount':'/home/francesco.pugnaloni/armadillo_all/tmp/embdi_caches/cache'+name+'/',
        'out_path_embedding_dict':'/home/francesco.pugnaloni/armadillo_all/datasets/WikiTables/dictionaries/embedding_dictionaries/embdi/embedding_dict'+name+'.pkl',
        'out_path_t_exec':'/home/francesco.pugnaloni/armadillo_all/datasets/WikiTables/dictionaries/embedding_dictionaries/embdi/t_execs'+name+'.pkl'
    }
    
    test = pd.read_csv('/home/francesco.pugnaloni/armadillo_all/datasets/WikiTables/test.csv')
    name = '_test'
    params_test = {
        'triple_files_list':[test],
        'table_dict':'/home/francesco.pugnaloni/armadillo_all/datasets/WikiTables/dictionaries/table_dict.pkl',
        'cache_directory_mount':'/home/francesco.pugnaloni/armadillo_all/tmp/embdi_caches/cache'+name+'/',
        'out_path_embedding_dict':'/home/francesco.pugnaloni/armadillo_all/datasets/WikiTables/dictionaries/embedding_dictionaries/embdi/embedding_dict'+name+'.pkl',
        'out_path_t_exec':'/home/francesco.pugnaloni/armadillo_all/datasets/WikiTables/dictionaries/embedding_dictionaries/embdi/t_execs'+name+'.pkl'
    }

    valid = pd.read_csv('/home/francesco.pugnaloni/armadillo_all/datasets/WikiTables/valid.csv')
    name = '_valid'
    params_valid = {
        'triple_files_list':[valid],
        'table_dict':'/home/francesco.pugnaloni/armadillo_all/datasets/WikiTables/dictionaries/table_dict.pkl',
        'cache_directory_mount':'/home/francesco.pugnaloni/armadillo_all/tmp/embdi_caches/cache'+name+'/',
        'out_path_embedding_dict':'/home/francesco.pugnaloni/armadillo_all/datasets/WikiTables/dictionaries/embedding_dictionaries/embdi/embedding_dict'+name+'.pkl',
        'out_path_t_exec':'/home/francesco.pugnaloni/armadillo_all/datasets/WikiTables/dictionaries/embedding_dictionaries/embdi/t_execs'+name+'.pkl'
    }

    # embed_triple_files_list(**params_t_0)
    # embed_triple_files_list(**params_t_1)
    # embed_triple_files_list(**params_t_2)
    # embed_triple_files_list(**params_t_3)
    # embed_triple_files_list(**params_t_4)
    # embed_triple_files_list(**params_t_5)
    # embed_triple_files_list(**params_t_6)
    # embed_triple_files_list(**params_t_7)
    # embed_triple_files_list(**params_t_8)
    # embed_triple_files_list(**params_t_9)
    # embed_triple_files_list(**params_test)
    embed_triple_files_list(**params_valid)