import pickle
import pandas as pd
import time
from tqdm import tqdm

def get_total_time(table_dict: str, df: str, outpath: str=None):
    with open(table_dict, 'rb') as f:
        table_dict = pickle.load(f)
    df = pd.read_csv(df)
    overlap_comp_times_jaccard = []
    start = time.time()
    for r in tqdm(range(df.shape[0])):
        start_jacc = time.time()
        r_t = table_dict[df.iloc[r]['r_id']]
        s_t = table_dict[df.iloc[r]['s_id']]

        r_t_s = set(r_t.values.reshape(-1))
        s_t_s = set(s_t.values.reshape(-1))

        jaccard = len(r_t_s.intersection(s_t_s)) / len(r_t_s.union(s_t_s))
        end_jacc = time.time()
        overlap_comp_times_jaccard.append(end_jacc-start_jacc)
    end = time.time()
    df['jaccard_runtime_correct'] = overlap_comp_times_jaccard
    if isinstance(outpath, str):
        df.to_csv(outpath, index=False)
    print(f'Jaccard runtime: {sum(overlap_comp_times_jaccard)}s')
    return (end-start)

def get_total_time_querying_optim(table_dict: str, df: str):
    with open(table_dict, 'rb') as f:
        table_dict = pickle.load(f)
    df = pd.read_csv(df)
    start = time.time()
    set_dict = {k:set(table_dict[k].values.reshape(-1)) for k in table_dict.keys()}
    for r in tqdm(range(df.shape[0])):
        r_t_s = set_dict[df.iloc[r]['r_id']]
        s_t_s = set_dict[df.iloc[r]['s_id']]

        jaccard = len(r_t_s.intersection(s_t_s)) / len(r_t_s.union(s_t_s))
    end = time.time()
    print(f'Jaccard optim runtime: {end-start}s')

if __name__=='__main__':

    get_total_time_querying_optim('/home/francesco.pugnaloni/armadillo_all/datasets/GitTables/table_querying/dictionaries/table_dict_table_querying.pkl', 
                                   '/home/francesco.pugnaloni/armadillo_all/datasets/GitTables/table_querying/table_querying_stats.csv')

    # time_wiki = get_total_time('/home/francesco.pugnaloni/armadillo_all/datasets/WikiTables/dictionaries/table_dict.pkl', 
    #                            '/home/francesco.pugnaloni/armadillo_all/datasets/WikiTables/test.csv',
    #                            '/home/francesco.pugnaloni/armadillo_all/datasets/WikiTables/test_with_correct_jaccard_time.csv'
    #                            )
    # print(f'total time WikiTables: {time_wiki}s')

    # time_git = get_total_time('/home/francesco.pugnaloni/armadillo_all/datasets/GitTables/dictionaries/table_dictionaries/table_dict.pkl',
    #                           '/home/francesco.pugnaloni/armadillo_all/datasets/GitTables/test.csv',
    #                           '/home/francesco.pugnaloni/armadillo_all/datasets/GitTables/test_with_correct_jaccard_time.csv')
    # print(f'total time GitTables: {time_git}s')

    # time_querying = get_total_time('/home/francesco.pugnaloni/armadillo_all/datasets/GitTables/table_querying/dictionaries/table_dict_table_querying.pkl', 
    #                                '/home/francesco.pugnaloni/armadillo_all/datasets/GitTables/table_querying/table_querying_stats.csv',
    #                                '/home/francesco.pugnaloni/armadillo_all/datasets/GitTables/table_querying/table_querying_stats_with_correct_jaccard_time.csv')
    # print(f'total time TableQuerying: {time_querying}s')


