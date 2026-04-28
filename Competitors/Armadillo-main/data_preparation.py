import os
from Code._csv_preprocessing import generate_table_dict_from_different_folders

if __name__ == '__main__':
    root = ''                                       # Path to the directory containing the datasets
    root_git = root+'/gittables/'                   
    root_wiki = root+'/wikitables/'
    root_table_querying = root_git

    print('Building table_dict for GitTables')
    if not os.path.exists(root_git+'/dictionaries/'):
        os.makedirs(root_git+'/dictionaries/')
    generate_table_dict_from_different_folders(
        [root_git+'/train/',root_git+'/test/',root_git+'/valid/'], 
        outpath=root_git+'/dictionaries/table_dict.pkl', anon=False)

    print('Building table_dict for WikiTables')
    if not os.path.exists(root_wiki+'/dictionaries/'):
        os.makedirs(root_wiki+'/dictionaries/')
    generate_table_dict_from_different_folders(
        [root_wiki+'/train/',root_wiki+'/test/',root_wiki+'/valid/'], 
        outpath=root_wiki+'/dictionaries/table_dict.pkl', anon=False)
    
    if not os.path.exists(root_table_querying+'/dictionaries/'):
        os.makedirs(root_table_querying+'/dictionaries/')
    generate_table_dict_from_different_folders(
        [root_table_querying+'/tables/'], 
        outpath=root_table_querying+'/dictionaries/table_dict_querying.pkl', anon=False)
    