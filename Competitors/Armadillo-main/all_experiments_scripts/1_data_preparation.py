import os
from Code._csv_preprocessing import generate_table_dict_from_different_folders

if __name__ == '__main__':
    root = ''                                       # Path to the directory containing the datasets
    root_git = root+'/GitTables/'                   
    root_wiki = root+'/WikiTables/'
    root_table_querying = root+'/Table_querying'

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
    
    print('Building noised table_dict for WikiTables')
    if not os.path.exists(root_wiki+'/dictionaries/'):
        os.makedirs(root_wiki+'/dictionaries/')
    generate_table_dict_from_different_folders(
        [root_wiki+'/train/',root_wiki+'/test/',root_wiki+'/valid/'], 
        outpath=root_wiki+'/dictionaries/table_dict_noised.pkl', anon=True)
    
    print('Building noised table_dict for table querying')
    if not os.path.exists(root_table_querying+'/dictionaries/'):
        os.makedirs(root_table_querying+'/dictionaries/')
    generate_table_dict_from_different_folders(
        [root_table_querying+'/tables/'], 
        outpath=root_table_querying+'/dictionaries/table_dict_noised.pkl', anon=True)
    