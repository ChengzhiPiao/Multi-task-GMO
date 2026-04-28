from _revision_ablation_study.random_node_embeddings_init.graph import *
from _revision_ablation_study.random_node_embeddings_init.node_embeddings import Random_Embedding_Buffer
from tqdm import tqdm
import sys
import time

def generate_graph_dictionary(table_dict_path: str | dict, out_path: str=None, embedding_generation_method: str='random', save_graph_dict: bool=True) -> dict:
    """Generate a graph dictionary from a table dictionary

    Args:
        table_dict_path (str): path to the table dictionary
        out_path (str): path to the file where to save the new graph dictionary
        embedding_generation_method (str, optional): approach used to generate embeddings, possible values are 'fasttext' and 'BERT'. Defaults to 'fasttext'
        save_graph_dict (bool, optional): if true the graph_dict will be dumped in the out_path. Deafaults to true
    Returns:
        dict: the generated graph dictionary
    """
    embedding_generation_method = 'random'
    start = time.time()
    print('Loading table_dict.....')
    if isinstance(table_dict_path, str):
        try:
            with open(table_dict_path,'rb') as f:
                table_dict = pickle.load(f)
        except:
            raise Exception("table_dict not found")
    else:
        table_dict = table_dict_path
    print('Table dict loaded')
    end = time.time()
    print(f'Table loaded in: {(end-start)}s\n')

    start_interm = time.time()
    print('Istantiating embedding buffer.....')
    embedding_buffer = Random_Embedding_Buffer()
    print('Embedding buffer instantiated')
    end = time.time()
    print(f'Embedding_buffer instantiated in: {(end-start_interm)}s\n')
    
    out = {}

    start_interm = time.time()
    print('Graphs generation starts.....')
    for k in tqdm(table_dict.keys()):
        try:
            out[k] = Graph(table_dict[k], k, merge_nodes_same_value=False, embedding_buffer=embedding_buffer)
        except:
            out[k] = None
    print('Graph generation ends')
    if save_graph_dict and (isinstance(out_path, str)):
        print('Saving output')
        with open(out_path, 'wb') as f:
            pickle.dump(out, f)   
        print('Output saved')
    end = time.time()
    print(f'Graph_dict generated in: {(end-start_interm)}s')
    print(f'Total t_exec: {(end-start)}s')
    return out