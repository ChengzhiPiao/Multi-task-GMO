'''
Code mainly taken and modified from: https://github.com/HPI-Information-Systems/prisma/blob/main/src/main/resources/embdi/embdi_wrapper.py
'''

import os
import sys
import torch
import argparse
import pandas as pd
import gensim.models as models
from numpy import dot
from gensim import matutils
from pathlib import Path
from embeddings import learn_embeddings
from embdi.EmbDI.graph import graph_generation
import embdi.EmbDI.utils as eutils
from embdi.EmbDI.data_preprocessing import data_preprocessing, write_info_file, get_unique_string_values
from embdi.EmbDI.edgelist import EdgeList
from embdi.EmbDI.sentence_generation_strategies import random_walks_generation
from embdi.EmbDI.utils import (TIME_FORMAT, read_edgelist)
from embdi.EmbDI.schema_matching import _produce_match_results , _extract_candidates
import numpy as np

REPO_DIR = '/home/francesco.pugnaloni/armadillo_all/Armadillo_local/'

# CACHE_DIRECTORY_MOUNT = REPO_DIR+"Baselines/embdi/embdi/cache"

# if not os.path.exists(CACHE_DIRECTORY_MOUNT):
#     os.makedirs(CACHE_DIRECTORY_MOUNT)

# INFO_FILE_FP = f"{CACHE_DIRECTORY_MOUNT}/info_file.csv"
# EDGELIST_FP = f"{CACHE_DIRECTORY_MOUNT}/edgelist"
# EMBEDDINGS_FP = f"{CACHE_DIRECTORY_MOUNT}/embeddings"

PREFIXES = ["3#__tn", "3$__tt", "5$__idx", "1$__cid"]

# default parameters for embdi
DEFAULT_PARAMS = {
    'ntop': 10,
    'ncand': 1,
    'max_rank': 3,
    'follow_sub': False,
    'smoothing_method': 'no',
    'backtrack': True,
    'training_algorithm': 'word2vec',
    'write_walks': True,
    'flatten': 'tt',
    'indexing': 'basic',
    'epsilon': 0.1,
    'num_trees': 250,
    'compression': False,
    'n_sentences': 'default',
    'walks_strategy': 'basic',
    'learning_method': 'skipgram',
    'sentence_length': 60,
    'window_size': '3',
    'n_dimensions': '300',
    'experiment_type': 'SM',
    'intersection': False,
    'walks_file': None,
    'mlflow': False,
    'repl_numbers': False,
    'repl_strings': False,
    'sampling_factor': 0.001,
    'output_file': 'small_example',
    'concatenate': 'horizon',
    'missing_value': 'nan,ukn,none,unknown,',
    'missing_value_strategy': '',
    'round_number': 0,
    'round_columns': 'price',
    'auto_merge': False,
    'tokenize_shared': False,
    'run-tag': 'something_random',
    'follow_replacement': False
}

def read_csv(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    return df

def write_csv(df: pd.DataFrame, file_path: str):
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(file_path, index=False)


def embeddings_generation(walks, dictionary, embeddings_file_name, params):
    """
    Take the generated walks and train embeddings using the walks as training corpus.
    :param walks:
    :param dictionary:
    :return:
    """
    learn_embeddings(embeddings_file_name, walks, write_walks=params['write_walks'],
                     dimensions=int(params['n_dimensions']),
                     window_size=int(params['window_size']))

    if params['compression']:
        newf = eutils.clean_embeddings_file(embeddings_file_name, dictionary)
    else:
        newf = embeddings_file_name
    params['embeddings_file'] = newf

    return params

def dot_product_similarity_matrix(wv, dataset, source_columns, target_columns):
    similarity_matrix = [[0.0 for _ in target_columns] for __ in source_columns]
    for i, source_column in enumerate(source_columns):
        for j, target_column in enumerate(target_columns):
            source_embedding_string, target_embedding_string = f"cid__{source_column}", f"cid__{target_column}"
            if source_embedding_string in wv and target_embedding_string in wv:
                similarity_matrix[i][j] = dot(matutils.unitvec(wv[source_embedding_string]), matutils.unitvec(wv[target_embedding_string]))

    return similarity_matrix

def binary_similarity_matrix_from_embdi(wv, dataset, source_columns, target_columns):
    candidates = _extract_candidates(wv, dataset)
    match_results = _produce_match_results(candidates)
    sm = [[0.0 for _ in target_columns] for __ in source_columns]

    i_emb_col_names = list(enumerate([f"0_{col}" for col in source_columns])) + list(enumerate([f"1_{col}" for col in target_columns]))

    lookup = {col: i for i,col in i_emb_col_names}

    for match_result in match_results:
        sm[lookup[match_result[0]]][lookup[match_result[1]]] = 1.0

    return sm

def prepare_csv(df_1, df_2, params):
    # overrides default values to schema matching values
    # might need adjustment for new datasets
    return data_preprocessing([df_1, df_2], params)

def generate_edgelist(df, info_file, EDGELIST_FP):
    return EdgeList(df, EDGELIST_FP, PREFIXES, info_file, flatten=True)

def generate_random_walks(params, EDGELIST_FP):
    prefixes, edgelist = read_edgelist(EDGELIST_FP)
    graph = graph_generation(params, edgelist, prefixes, dictionary=None)
    #  Compute the number of sentences according to the rule of thumb.
    if params['n_sentences'] == 'default':
        params['n_sentences'] = graph.compute_n_sentences(int(params['sentence_length']))
    params["write_walks"] = False
    walks = random_walks_generation(params, graph)

    return walks

def table_identifier(csv_path):

    #return f"{dataset}_{table}"
    return csv_path

def filter_embeddings(embeddings_path):
    with open(embeddings_path, "r") as emb_f:
        unfiltered = emb_f.readlines()
    dimension = unfiltered[0].split(" ")[1].strip("\n")
    filtered = [line for line in unfiltered if line.startswith("cid__")]
    # also add embedding without the cid__:
    non_prefixed = [line[5:] for line in filtered]
    with open(embeddings_path, "w") as emb_f:
        emb_f.write(f"{len(filtered)+ len(non_prefixed)} {dimension}\n")
        emb_f.writelines(filtered)
        emb_f.writelines(non_prefixed)


def read_variables_file(var_file):
    variables = {}
    with open(var_file, 'r') as fp:
        for i, line in enumerate(fp):
            parameter, values = line.strip().split(':', maxsplit=1)
            try:
                values = eval(values)
            except:
                pass
            variables[parameter] = values
    return variables

def update_params(scenario_path, params, verbose=False):
    if verbose:
        print(params, flush=True)

    config = read_variables_file(scenario_path)
    if verbose:
        print(" ----  Loaded variable file ---- ", flush=True)
    for k,v in config.items():
        params[k] = v
    if verbose:
        print(params, flush=True)
    return params




def import_database(database_folder):
    dfs = []
    files = sorted([file for file in os.listdir(database_folder) if file.endswith(".csv")])
    for file in files:
        dfs.append(read_csv(os.path.join(database_folder, file)))
    concat = pd.concat(dfs, axis=1)
    concat.columns = [str(id).zfill(5) + c for id, c in enumerate(concat.columns)]
    return concat

def import_scenario(scenario_path):
    source_df = import_database(os.path.join(os.sep + scenario_path, "source"))
    target_df = import_database(os.path.join(os.sep + scenario_path, "target"))
    return source_df, target_df

def get_emb_names(df: pd.DataFrame, n_cols_source, n_rows_source, n_cols_target, n_rows_target):
    n_rows_target = n_rows_source + n_rows_target
    n_cols_target = n_cols_source + n_cols_target
    set_src = [f'idx__{i}' for i in range(0,n_rows_source)]
    set_target = [f'idx__{i}' for i in range(n_rows_source, n_rows_target)]
    cols = list(df.columns)
    for c in range(n_cols_source):
        set_src.append(f'cid__{cols[c]}')
    for r in range(n_rows_source):
        for c in range(n_cols_source):
            set_src.append(df.iloc[r].iloc[c])

    for c in range(n_cols_source, n_cols_target):
        set_target.append(f'cid__{cols[c]}')
    for r in range(n_rows_source, n_rows_target):
        for c in range(n_cols_source, n_cols_target):
            set_target.append(df.iloc[r].iloc[c])
    return set(set_src), set(set_target)

def aggregate_embeddings(wv, value_set) -> torch.Tensor:
    embeddings = []
    pref_1 = 'tt__'
    pref_2 = 'tn__'
    for k in value_set:
        try:
            embeddings.append(wv[pref_1+k])
        except:
            try:
                embeddings.append(wv[pref_2+k])
            except:         
                try:
                    embeddings.append(wv[k])
                except:
                    None
    if len(embeddings) == 0:
        return torch.zeros(300)
    else:
        return torch.Tensor(sum(embeddings)/len(embeddings))

def generate_table_embeddings(scenario_path, df_1, df_2, cache_directory_mount=None):
    if cache_directory_mount == None:
        CACHE_DIRECTORY_MOUNT = REPO_DIR+"Baselines/embdi/embdi/cache"
    else:
        CACHE_DIRECTORY_MOUNT = cache_directory_mount

    if not os.path.exists(CACHE_DIRECTORY_MOUNT):
        os.makedirs(CACHE_DIRECTORY_MOUNT)

    INFO_FILE_FP = f"{CACHE_DIRECTORY_MOUNT}/info_file.csv"
    EDGELIST_FP = f"{CACHE_DIRECTORY_MOUNT}/edgelist"
    EMBEDDINGS_FP = f"{CACHE_DIRECTORY_MOUNT}/embeddings"
    execution_specific_params = update_params(scenario_path, DEFAULT_PARAMS.copy())
    if isinstance(df_1, str):
        df_1 = pd.read_csv(df_1)
    if isinstance(df_2, str):
        df_2 = pd.read_csv(df_2)
    input_1 = "_source"
    input_2 = "_target"
    try:
        execution_specific_params["expand_columns"] = ','.join(list(set(list(df_1.columns) + list(df_2.columns))))
    except:
        c1_list = [str(v) for v in df_1.columns]
        c2_list = [str(v) for v in df_2.columns]
        df_1.columns = c1_list
        df_2.columns = c2_list
        execution_specific_params["expand_columns"] = ','.join(list(set(list(df_1.columns) + list(df_2.columns))))
    preprocessed = prepare_csv(df_1, df_2, execution_specific_params)
    df_1_value_set, df_2_value_set = get_emb_names(preprocessed, df_1.shape[1], df_1.shape[0], df_2.shape[1], df_2.shape[0])
    Path(INFO_FILE_FP).parent.mkdir(parents=True, exist_ok=True)
    write_info_file([df_1, df_2], INFO_FILE_FP, [input_1, input_2])
    edgelist = generate_edgelist(preprocessed, INFO_FILE_FP, EDGELIST_FP=EDGELIST_FP)
    try: 
        walks = generate_random_walks(execution_specific_params, EDGELIST_FP=EDGELIST_FP)
    except:
        return torch.zeros(300), torch.ones(300)
    embeddings_generation(walks, None, EMBEDDINGS_FP, execution_specific_params)

    wv = models.KeyedVectors.load_word2vec_format(EMBEDDINGS_FP, unicode_errors='ignore')

    df_1_emb = aggregate_embeddings(wv, df_1_value_set)
    df_2_emb = aggregate_embeddings(wv, df_2_value_set)

    return df_1_emb, df_2_emb



if __name__ == "__main__":
    # args = parse_args()
    # print(match(args.input_1, args.input_2))
    generate_table_embeddings('/home/francesco.pugnaloni/armadillo_all/Armadillo_local/Baselines/embdi/config-dblp_acm-sm', '/home/francesco.pugnaloni/armadillo_all/datasets/WikiTables/test/tables/373.87126.csv', '/home/francesco.pugnaloni/armadillo_all/datasets/WikiTables/test/tables/462.44160.csv')