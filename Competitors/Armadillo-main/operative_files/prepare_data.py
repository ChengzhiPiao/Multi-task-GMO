import os
import pandas as pd
from Code._csv_preprocessing import *

def get_table_list(df: pd.DataFrame) -> list:
    out = []
    for r in tqdm(range(df.shape[0])):
        out.append(df.iloc[r]['r_id'])
        out.append(df.iloc[r]['s_id'])
    return out
def get_table_set(tables: list) -> set:
    out = []
    for t in tables:
        out += get_table_list(t)
    return set(out)

def generate_sets_table_querying(df: pd.DataFrame, root: str) -> None:
    query_set = []
    data_lake = []
    for r in tqdm(range(df.shape[0])):
        query_set.append(df.iloc[r]['r_id'])
        data_lake.append(df.iloc[r]['s_id'])
    query_set = set(query_set)
    data_lake = set(data_lake)
    with open(root+'/query_set_1k.pkl', 'wb') as f:
        pickle.dump(query_set, f)
    with open(root+'/data_lake_10k.pkl', 'wb') as f:
        pickle.dump(data_lake, f)

if __name__ == '__main__':
    """
        Input: 
            * root directories for gittables and wikilast containing:
                - train.csv
                - test.csv
                - valid.csv
            * directories containing all the csv files in wikilast and gittables
        Output:
            * inside the root directories are created:
                - 2x table_dict.pkl
                - queryt_set_1k.pkl
                - data_lake_10k.pkl
    """
    root = '/home/francesco.pugnaloni/armadillo_all/datasets/'                       # Insert the full name of the root directory here
    gittables_csv_directory = root+'/GitTables/'
    # wikitables_csv_directory = root+'/WikiTables/'

    print('Building table_dict for GitTables')
    if not os.path.exists(gittables_csv_directory+'/dictionaries/'):
        os.makedirs(gittables_csv_directory+'/dictionaries/')
    # generate_table_dict_from_different_folders(
    #     [gittables_csv_directory+'/train/',gittables_csv_directory+'/test/',gittables_csv_directory+'/valid/'], 
    #     outpath=gittables_csv_directory+'/dictionaries/table_dict.pkl', anon=False)
    
    # generate_table_dict_from_different_folders(
    #     [gittables_csv_directory+'/train/',gittables_csv_directory+'/test/',gittables_csv_directory+'/valid/'], 
    #     outpath=gittables_csv_directory+'/dictionaries/table_dict_anon.pkl', anon=True)

    # generate_table_dict_from_different_folders(
    #     ['/home/francesco.pugnaloni/armadillo_all/datasets/GitTables/table_querying/'], 
    #     outpath='/home/francesco.pugnaloni/armadillo_all/datasets/GitTables/table_querying/dictionaries/table_dict_table_querying.pkl', anon=False)
    
    print('Building anonymized table_dict for table querying')
    generate_table_dict_from_different_folders(
        ['/home/francesco.pugnaloni/armadillo_all/datasets/GitTables/table_querying/table_data/'], 
        outpath='/home/francesco.pugnaloni/armadillo_all/datasets/GitTables/table_querying/dictionaries/table_dict_table_querying_anon.pkl', anon=True)
    
    # print('Building table_dict for WikiTables')
    # if not os.path.exists(wikitables_csv_directory+'/dictionaries/'):
    #     os.makedirs(wikitables_csv_directory+'/dictionaries/')
    
    # print('Building anonymized table_dict for WikiTables')
    # generate_table_dict_from_different_folders(
    #     [wikitables_csv_directory+'/train/',wikitables_csv_directory+'/test/',wikitables_csv_directory+'/valid/'], 
    #     outpath=wikitables_csv_directory+'/dictionaries/table_dict_anon.pkl', anon=True)
    
    print('Data preparation is complete')
