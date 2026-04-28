import pandas as pd
import pickle
from tqdm import tqdm

def get_missing_tables(query_pairs: str|pd.DataFrame, table_dict: str | dict):
    if isinstance(query_pairs, str):
        query_pairs = pd.read_csv(query_pairs)
    if isinstance(table_dict, str):
        with open(table_dict, 'rb') as f:
            table_dict = pickle.load(f)
    tables_set_dict = set(table_dict.keys())
    tables_set_ndcg = set(query_pairs['r_id']).union(set(query_pairs['s_id']))
    missing_tables = set(t for t in tables_set_ndcg if t not in tables_set_dict)
    none_embs = []
    for t in tqdm(tables_set_ndcg):
        try:
            table_dict[t]
            if t == None:
                none_embs.append(t)
        except:
            pass
    return missing_tables, set(none_embs)

if __name__ == '__main__':
    query_dataset = '/home/francesco.pugnaloni/armadillo_all/datasets/GitTables/table_querying/table_querying_stats.csv'
    embedding_dataset = '/home/francesco.pugnaloni/armadillo_all/datasets/GitTables/dictionaries/embedding_dictionaries/emb_dict_turl_tables_128_128.pkl'
    missing, none_embds = get_missing_tables(query_dataset,embedding_dataset)
    pass