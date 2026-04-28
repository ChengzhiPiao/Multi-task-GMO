import sys
sys.path.append(".")
sys.path.append("../../")
sys.path.append("../../..")
sys.path.append("../../../..")
import pickle
import pandas as pd
from tqdm import tqdm
import os
from Baselines.gen_t.gen_t.code.findCandidates.set_similarity import main as set_similarity_main
from Baselines.gen_t.gen_t.code.discovery.recover_matrix_ternary import main as recover_matrix_ternary_main
from Baselines.gen_t.gen_t.code.integration.table_integration import main as table_integration_main
from Baselines.gen_t.gen_t.code.evaluatePaths import setTDR, bestMatchingTuples, instanceSimilarity
from Baselines.gen_t.constants import *
import clevercsv
from Code._csv_preprocessing import *
from Code._generate_graph_dict import *
import pickle
from Code.armadillo import *
import time
import torch
from Code._table_querying import *
from Code._embed_all_no_paral import run_experiment

class GraphDataset(Dataset):
    
    def __init__(self, triples: pd.DataFrame, graphs: dict, tables: Optional[dict] = None) -> None:
        """Init function

        Args:
            triples (pd.DataFrame): Dataframe that contains triples ()'r_id','s_id','table_overlap')
            graphs (dict): a dictionary containing a graph for every key that appears in the triples dataset
            tables (Optional[dict], optional): not implemented. Defaults to None.
        """
        super(GraphDataset, self).__init__()
        self.graphs = graphs
        self.keys = graphs.keys()

    def len(self) -> int:
        return len(self.keys)
    
    def get(self, idx:int) -> tuple:
        k = self.keys[idx]
        try:
            g1 = self.graphs[k]
        except:
            g1 = self.graphs[str(int(k))]
        return Data(g1.X, g1.edges), k

def replace_nans(df: pd.DataFrame) -> pd.DataFrame:
    for r in range(df.shape[0]):
        for c in range(df.shape[1]):
            if pd.isna(df.iloc[r,c]):
                df.iloc[r,c] = 'n.a.n'
    return df

def find_pk(df: pd.DataFrame) -> pd.DataFrame:
    max_c = -1
    col = 0
    for c in range(df.shape[1]):
        curr_c = len(set(df.iloc[:,c]))
        if curr_c > max_c:
            max_c = curr_c
            col = c
    to_drop = []
    vals = {}
    for r in range(df.shape[0]):
        v = df.iloc[r,col]
        try:
            vals[v]
            to_drop.append(r)
        except:
            vals[v] = 0
    df = df.drop(to_drop)
    df = pd.concat([df.iloc[:,col], df.drop(col, axis=1)], axis=1)
    return df

def generate_embedding_dict(graph_dict: dict, model_file: str) -> dict:
    model = Armadillo(model_file=model_file)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embedding_dict = {}
    with torch.no_grad():
        for k in tqdm(graph_dict.keys()):
            try:
                embedding_dict[k] = model(graph_dict[k]['X'])
            except:
                print(f'Error embedding {k}')

    return embedding_dict

def embed_all_armadillo(datalake_dict: str, query_tables_path: str, model_file: str, out_path: str=None, out_t_execs: str=None) -> dict:
    print('Generating table dictionary')
    table_dict = {k:datalake_dict[k].iloc[:160,:13] for k in datalake_dict.keys()}
    _, query_dict = load_data_lake_dict(query_tables_path)
    query_tables = set(query_dict.keys())
    target_tables = set(datalake_dict.keys())
    
    start = time.time()
    print('Generating graph dictionary')
    graph_dict = generate_graph_dictionary(table_dict, embedding_generation_method='sha256', save_graph_dict=False)
    print(f'Cleaning graph dict, old n_elements: {len(graph_dict)}')
    graph_dict = {k:graph_dict[k] for k in graph_dict.keys() if graph_dict[k] is not None}
    print(f'new len: {len(graph_dict)}')

    print('Generating embeddings')
    embedding_dict = run_experiment(model_file=model_file, table_dict_path=table_dict, graphs_path=graph_dict)
    target_tables = set(graph_dict.keys())
    index_to_table_mapping, embedding_tensor = build_embedding_tensor(target_set=target_tables, target_embedding_dict=embedding_dict)
        
    model = Armadillo(model_file=model_file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}
    print('Armadillo querying running')
    for t_query in tqdm(query_tables):
        results[t_query] = table_querying(t_query=t_query, target_set=target_tables, model=model, table_dict=query_dict, target_embedding_tensor=embedding_tensor, index_to_table_mapping=index_to_table_mapping)
    print('Building output dataframe blocking')
    out_df = {'r_id':[], 's_id':[], 'overlap_ratio':[]}
    for r_id in tqdm(query_tables):
        for s_id, overlap in results[r_id]['overlaps']:
            out_df['r_id'].append(r_id)
            out_df['s_id'].append(s_id)
            out_df['overlap_ratio'].append(max(0, overlap))
    out_df = pd.DataFrame(out_df)
    end = time.time()
    results['total_time_with_embeddings'] = end-start
    if isinstance(out_path, str):
        out_df.to_csv(out_path, index=False)
    total_time = end-start
    print(f'Total time armadillo blocking: {total_time}')
    # save the total_time in a .txt file in out_t_execs
    if isinstance(out_t_execs, str):
        with open(out_t_execs, 'w') as file:
            file.write(str(total_time)+'s\n')
    return out_df

def create_lake_dict(DATALAKE_PATH: str) -> dict:
    lake_files = list(os.listdir(DATALAKE_PATH))
    data_lake_dict = {}
    data_lake_dict_raw = {}
    for f in tqdm(lake_files):
        try:
            df = pd.read_csv(DATALAKE_PATH +'/'+f)
        except:
            print(f'Error reading {f}')
            continue
        tmp = {}
        for c in df.columns:
            tmp[c] = df[c].tolist()
        data_lake_dict[f] = tmp
        data_lake_dict_raw[f] = df
    return data_lake_dict, data_lake_dict_raw

def load_data_lake_dict(data_lake_path: str=None, dict_path:str=None, dict_path_raw:str=None) -> tuple:
    try:
        with open(dict_path, 'rb') as file:
            data_lake_dict = pickle.load(file)
        with open(dict_path_raw, 'rb') as file:
            data_lake_dict_raw = pickle.load(file)
        return data_lake_dict, data_lake_dict_raw
    except:
        data_lake_dict, data_lake_dict_raw = create_lake_dict(data_lake_path)
    if isinstance(dict_path, str):
        with open(dict_path, 'wb') as file:
            pickle.dump(data_lake_dict, file)
    if isinstance(dict_path_raw, str):
        with open(dict_path_raw, 'wb') as file:
            pickle.dump(data_lake_dict_raw, file)
    return data_lake_dict, data_lake_dict_raw

def retrieve_generating_tables(query_tables_path: str|dict, query_table_name: str, data_lake_dict: dict|str, data_lake_dict_raw: dict|str, 
                               sim_threshold: float=0.5, armadillo_overlaps: pd.DataFrame=None, armadillo_threshold: float=0.2)-> set:
    if not(armadillo_overlaps is None):
        armadillo_overlaps = armadillo_overlaps[armadillo_overlaps['r_id']==query_table_name]
        armadillo_overlaps = armadillo_overlaps[armadillo_overlaps['overlap_ratio']>=armadillo_threshold]
        blocked_tables = set(armadillo_overlaps['s_id'])
        data_lake_dict = {k:data_lake_dict[k] for k in blocked_tables}
        data_lake_dict_raw = {k:data_lake_dict_raw[k] for k in blocked_tables}

    candidateTablesFound, noCandidates = set_similarity_main(query_tables_path=query_tables_path, sourceTableName=query_table_name, sim_threshold=sim_threshold, LakeDfs_processed=data_lake_dict, rawLakeDfs=data_lake_dict_raw, allLakeTableCols=None)
    # candidate_names = list(candidateTablesFound.keys())
    generating_tables, time_stats = recover_matrix_ternary_main(sourceTableName=query_table_name, query_tables_path=query_tables_path,
                                table_dict_raw=data_lake_dict_raw, candidate_tables_dict=candidateTablesFound)
    if generating_tables is not None:
        originating_tables_dict = {k:candidateTablesFound[k] for k in generating_tables}
        timed_out, noCandidates, numOutputVals, reclaimed_table = table_integration_main(datalake_raw_dict=data_lake_dict_raw, query_tables_path=query_tables_path, originatingTablesDict=originating_tables_dict, sourceTableName=query_table_name, timeout=100, outputPath='/home/francesco.pugnaloni/tmp/')
    else:
        originating_tables_dict = {}
        reclaimed_table = pd.DataFrame({k:{} for k in pd.read_csv(query_tables_path+'/'+query_table_name).columns})
    try:
        recall, precision = setTDR(queryDf=pd.read_csv(query_tables_path+'/'+query_table_name), integratedDf=reclaimed_table)
    except ZeroDivisionError:
        recall, precision = float('nan'), float('nan')
    return generating_tables, precision, recall

def reclaim_multiple_tables(root_dataset: str, use_armadillo_blocking: bool=True, armadillo_threshold: float=0.2) -> None:
    data_lake_path = root_dataset+'/'+'datalake/'
    query_tables_path = root_dataset+'/'+'queries/'
    data_lake_dict_path = root_dataset+'/'+'datalake_dict.pkl'
    data_lake_dict_raw_path = root_dataset+'/'+'datalake_dict_raw.pkl'
    outdir = root_dataset+'/'+'output/'
    out_t_execs_armadillo = outdir+'/armadillo_exec_time.txt'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    tmp_dir = root_dataset+'/'+'tmp/'
    if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)
    if use_armadillo_blocking:
        armadillo_overlaps_path = outdir+'/armadillo_overlaps.csv'    
    else:
        armadillo_overlaps_path = None
        armadillo_overlaps = None
    data_lake_dict, data_lake_dict_raw = load_data_lake_dict(data_lake_path, dict_path=data_lake_dict_path, dict_path_raw=data_lake_dict_raw_path)
    if isinstance(armadillo_overlaps_path, str) and use_armadillo_blocking==True:
        try:
            armadillo_overlaps = pd.read_csv(armadillo_overlaps_path)
        except:
            armadillo_overlaps = embed_all_armadillo(data_lake_dict_raw, query_tables_path, '/home/francesco.pugnaloni/armadillo_all/datasets/GitTables/models/armadillo_git.pth', out_path=armadillo_overlaps_path,
                                                     out_t_execs=out_t_execs_armadillo)
    reclamation_performance = {'query_table':[], 'precision':[], 'recall':[]}
    start = time.time()
    print('Starting table reclamation with gen-t')
    for f in tqdm(os.listdir(query_tables_path)):
        if not f.endswith('.csv'):
            continue
        generating_tables, precision, recall = retrieve_generating_tables(query_tables_path=query_tables_path, query_table_name=f,data_lake_dict=data_lake_dict,data_lake_dict_raw=data_lake_dict_raw, 
                                                          sim_threshold=0.5, armadillo_overlaps=armadillo_overlaps, armadillo_threshold=armadillo_threshold)
        reclamation_performance['query_table'].append(f)
        reclamation_performance['precision'].append(precision)
        reclamation_performance['recall'].append(recall)
    end = time.time()
    gen_t_time = end-start
    print(f'Total time gen-t: {gen_t_time}')
    if use_armadillo_blocking:
        with open(outdir+'/gen_t_exec_time_armadillo_blocking.txt', 'w') as file:
            file.write(str(gen_t_time)+'s\n')
    reclamation_performance = pd.DataFrame(reclamation_performance)
    if use_armadillo_blocking:
        reclamation_performance.to_csv(outdir+'/reclamation_performance_post_armadillo_blocking.csv', index=False)
    if not use_armadillo_blocking:
        reclamation_performance.to_csv(outdir+'/reclamation_performance_only_gen_t.csv', index=False)


def retrieve_generating_tables_no_reclamation(query_tables_path: str|dict, query_table_name: str, data_lake_dict: dict|str, data_lake_dict_raw: dict|str, 
                               sim_threshold: float=0.5)-> set:
    candidateTablesFound, noCandidates = set_similarity_main(query_tables_path=query_tables_path, sourceTableName=query_table_name, sim_threshold=sim_threshold, LakeDfs_processed=data_lake_dict, rawLakeDfs=data_lake_dict_raw, allLakeTableCols=None)
    generating_tables, time_stats = recover_matrix_ternary_main(sourceTableName=query_table_name, query_tables_path=query_tables_path,
                                table_dict_raw=data_lake_dict_raw, candidate_tables_dict=candidateTablesFound)
    if generating_tables is None:
        generating_tables = []
    return generating_tables

def retrieve_multiple_generating_tables(root_dataset_armadillo: str, root_dataset_gen_t: str, armadillo_threshold: float=0.2, gen_t_threshold: float=0.5, embeddings_only: bool=False, use_armadillo_wiki: bool=False) -> None:

    if use_armadillo_wiki:
        data_lake_path_armadillo = root_dataset_armadillo+'/'+'datalake/'
        query_tables_path_armadillo = root_dataset_armadillo+'/'+'queries/'
        data_lake_dict_path_armadillo = root_dataset_armadillo+'/'+'datalake_dict.pkl'
        data_lake_dict_raw_path_armadillo = root_dataset_armadillo+'/'+'datalake_dict_raw.pkl'
        outdir_armadillo = root_dataset_armadillo+'/'+'output/'
        out_t_execs_armadillo = outdir_armadillo+'/armadillo_exec_time_armadillo_wiki.txt'
        out_performance = outdir_armadillo+f'/performance_arm_{armadillo_threshold}_wiki_gen_t_{gen_t_threshold}.csv'
        out_performance_raw = outdir_armadillo+f'/performance_arm_{armadillo_threshold}_wiki_gen_t_{gen_t_threshold}_raw.pkl'
        armadillo_overlaps_path = outdir_armadillo+'/armadillo_wiki_overlaps.csv' 
    else:
        data_lake_path_armadillo = root_dataset_armadillo+'/'+'datalake/'
        query_tables_path_armadillo = root_dataset_armadillo+'/'+'queries/'
        data_lake_dict_path_armadillo = root_dataset_armadillo+'/'+'datalake_dict.pkl'
        data_lake_dict_raw_path_armadillo = root_dataset_armadillo+'/'+'datalake_dict_raw.pkl'
        outdir_armadillo = root_dataset_armadillo+'/'+'output/'
        out_t_execs_armadillo = outdir_armadillo+'/armadillo_exec_time.txt'
        out_performance = outdir_armadillo+f'/performance_arm_{armadillo_threshold}_gen_t_{gen_t_threshold}.csv'
        out_performance_raw = outdir_armadillo+f'/performance_arm_{armadillo_threshold}_gen_t_{gen_t_threshold}_raw.pkl'
        armadillo_overlaps_path = outdir_armadillo+'/armadillo_overlaps.csv' 

    if not os.path.exists(outdir_armadillo):
        os.makedirs(outdir_armadillo)
    tmp_dir = root_dataset_armadillo+'/'+'tmp/'
    if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)

    data_lake_dict, data_lake_dict_raw = load_data_lake_dict(data_lake_path_armadillo, dict_path=data_lake_dict_path_armadillo, dict_path_raw=data_lake_dict_raw_path_armadillo)
    data_lake_size = len(data_lake_dict)
    try:
        armadillo_overlaps = pd.read_csv(armadillo_overlaps_path)
    except:
        if use_armadillo_wiki:
            armadillo_overlaps = embed_all_armadillo(data_lake_dict_raw, query_tables_path_armadillo, '/home/francesco.pugnaloni/armadillo_all/ablation_study/base/models/armadillo.pth', out_path=armadillo_overlaps_path,
                                                        out_t_execs=out_t_execs_armadillo)
        else:
            armadillo_overlaps = embed_all_armadillo(data_lake_dict_raw, query_tables_path_armadillo, '/home/francesco.pugnaloni/armadillo_all/datasets/GitTables/models/armadillo_git.pth', out_path=armadillo_overlaps_path,
                                                        out_t_execs=out_t_execs_armadillo)
        if embeddings_only:
            return

    data_lake_path_gen_t = root_dataset_gen_t+'/'+'datalake/'
    query_tables_path_gen_t = root_dataset_gen_t+'/'+'queries/'
    data_lake_dict_path_gen_t = root_dataset_gen_t+'/'+'datalake_dict.pkl'
    data_lake_dict_raw_path_gen_t = root_dataset_gen_t+'/'+'datalake_dict_raw.pkl'
    outdir_gen_t = root_dataset_gen_t+'/'+'output/'
    reclaimed_tables_path_gen_t = outdir_gen_t+'/reclaimed_tables_gen_t_only_{gen_t_threshold}.pkl'
    out_t_execs_gen_t = outdir_gen_t+f'/gen_t_exec_time_{gen_t_threshold}.txt'
    if not os.path.exists(outdir_gen_t):
        os.makedirs(outdir_gen_t)
    tmp_dir_gen_t = root_dataset_gen_t+'/'+'tmp/'
    if not os.path.exists(tmp_dir_gen_t):
            os.makedirs(tmp_dir_gen_t)
    data_lake_dict, data_lake_dict_raw = load_data_lake_dict(data_lake_path_gen_t, dict_path=data_lake_dict_path_gen_t, dict_path_raw=data_lake_dict_raw_path_gen_t)

    
    query_lake_size = len(os.listdir(query_tables_path_gen_t))

    reclaimed_tables_arm_gen_t = {}
    out_df_performance_arm_gen_t = {'query_table':[], 'pair_completeness':[], 'reduction_ratio':[]}

    try:
        with open(reclaimed_tables_path_gen_t, 'rb') as f:
            generating_tables_gen_t_backup = pickle.load(f)
    except:
        generating_tables_gen_t_backup = {}
    start = time.time()
    for f in tqdm(os.listdir(query_tables_path_gen_t)):
        if not f.endswith('.csv'):
            continue
        reclaimed_tables_arm_gen_t[f] = {'armadillo':[], 'gen_t':[]}
        tmp = armadillo_overlaps[armadillo_overlaps['r_id']==f]
        tmp = tmp[tmp['overlap_ratio']>=armadillo_threshold]
        blocked_tables_armadillo = set(tmp['s_id'])
        reclaimed_tables_arm_gen_t[f]['armadillo'] = blocked_tables_armadillo

        try:
            generating_tables = generating_tables_gen_t_backup[f]
        except:
            generating_tables = set(retrieve_generating_tables_no_reclamation(query_tables_path=query_tables_path_gen_t, query_table_name=f,data_lake_dict=data_lake_dict,data_lake_dict_raw=data_lake_dict_raw, 
                                                            sim_threshold=gen_t_threshold))
            generating_tables_gen_t_backup[f] = generating_tables

        reclaimed_tables_arm_gen_t[f]['gen_t'] = set(generating_tables)
        if len(generating_tables) == 0:
            pair_completeness = 1
        else:
            pair_completeness = len(generating_tables.intersection(blocked_tables_armadillo)) / len(generating_tables)
        reduction_ratio = 1-(len(blocked_tables_armadillo) / data_lake_size)

        reclaimed_tables_arm_gen_t[f]['pair_completeness'] = pair_completeness
        reclaimed_tables_arm_gen_t[f]['reduction_ratio'] = reduction_ratio

        out_df_performance_arm_gen_t['query_table'].append(f)
        out_df_performance_arm_gen_t['pair_completeness'].append(pair_completeness)
        out_df_performance_arm_gen_t['reduction_ratio'].append(reduction_ratio)
    end = time.time()
    if not os.path.exists(out_t_execs_gen_t):
        with open(out_t_execs_gen_t, 'w') as file:
            file.write(str(end-start)+'s\n')
        
    out_df_performance_arm_gen_t = pd.DataFrame(out_df_performance_arm_gen_t)
    print(f"Mean pair completeness: {out_df_performance_arm_gen_t['pair_completeness'].mean()} Mean reduction ratio: {out_df_performance_arm_gen_t['reduction_ratio'].mean()}")
    out_df_performance_arm_gen_t.to_csv(out_performance, index=False)
    with open(out_performance_raw, 'wb') as file:
        pickle.dump(reclaimed_tables_arm_gen_t, file)
    
    # if the file reclaimed_tables_path_gen_t not exists, create it
    if not os.path.exists(reclaimed_tables_path_gen_t):
        with open(reclaimed_tables_path_gen_t, 'wb') as file:
            pickle.dump(generating_tables_gen_t_backup, file)

    pass


if __name__ == '__main__':
    params_wiki_tptr_small = {
        'root_dataset_armadillo':'/home/francesco.pugnaloni/armadillo_all/gen-t/wiki_tptr_small/',
        'root_dataset_gen_t':'/home/francesco.pugnaloni/armadillo_all/gen-t/tptr_small/tptr_small/',
        'armadillo_threshold':0.01,
        'gen_t_threshold':0.5,
        'embeddings_only':False,
        'use_armadillo_wiki': True
    }
    retrieve_multiple_generating_tables(**params_wiki_tptr_small)