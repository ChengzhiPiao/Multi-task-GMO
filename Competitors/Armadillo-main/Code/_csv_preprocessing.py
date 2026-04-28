import pickle
import os
import pandas as pd
from tqdm import tqdm
import hashlib

def list_files(directory) -> list:
    """given the path of a directory return the list of its files

    Args:
        directory (_type_): path to the directory to explore

    Returns:
        list: list of filenames
    """
    l=[]
    with os.scandir(directory) as entries:
        for entry in entries:
            if entry.is_file():
                l.append(entry.name)
    return l

def generate_table_dict_old(csv_folder_path: str, outpath: str=None, table_set: set=None, log_path: str=None) -> dict:
    """_summary_

    Args:
        gittables_folder_path (str): path to the folder containing the tables in the csv format
        outpath (str): folder where to save the generated table_dict
        log_path (str, optional): path to the directory where to save the names of the dropped tables. Defaults to None.

    Returns:
        dict: a table_dict
    """
    filenames = list_files(csv_folder_path)
    log = []
    table_dict = {}
    for i in tqdm(range(len(filenames))):
        t = None
        if isinstance(table_set, set):
            if filenames[i] not in table_set:
                continue
        path = csv_folder_path + '/' + filenames[i]
        try:
            t = pd.read_csv(path, sep=',', dtype=str)
        except:
            try:
                t = pd.read_csv(path, sep='#', dtype=str)
            except:
                log.append(path)
        if isinstance(t, pd.DataFrame):
            table_dict[str(filenames[i])] = t
        
    if log_path:
        with open(log_path, 'w') as file:
            # Write the string to the file
            file.write('\n'.join(log))

    if isinstance(outpath, str):
        with open(outpath, 'wb') as f:
            pickle.dump(table_dict, f)
    return table_dict

def anonymize_tab(df: pd.DataFrame) -> pd.DataFrame:
    cols = {}
    for k in df.columns:
        cols[str(k)] = [hashlib.sha256(str(val).encode()).hexdigest() for val in df[k]]
    return pd.DataFrame(cols)

def generate_table_dict(collection_directory: str, outpath: str=None, anon:bool=False) -> dict:
    metadata_path = collection_directory+'/'+'metadata.csv'
    metadata = pd.read_csv(metadata_path)
    csv_path = collection_directory+'/'+'tables/'
    out_dict = {}
    for r in tqdm(range(metadata.shape[0])):
        meta = metadata.iloc[r]
        t_name = meta.loc['_id']
        n_header = meta.loc['num_header_rows']
        if anon:
            out_dict[t_name] = anonymize_tab(pd.read_csv(csv_path+'/'+t_name, dtype=str, header=None, skiprows=n_header))
        else:
            out_dict[t_name] = pd.read_csv(csv_path+'/'+t_name, dtype=str, header=None, skiprows=n_header)
    if isinstance(outpath, str):
        with open(outpath,'wb') as f:
            pickle.dump(out_dict, f)
    return out_dict
    
def generate_table_dict_from_different_folders(folders: list, outpath: str=None, anon:bool=False) -> dict:
    table_dict = {}
    for d in folders:
        table_dict.update(generate_table_dict(d, anon=anon))
    if isinstance(outpath, str):
        with open(outpath, 'wb') as f:
            pickle.dump(table_dict, f)
    return table_dict
