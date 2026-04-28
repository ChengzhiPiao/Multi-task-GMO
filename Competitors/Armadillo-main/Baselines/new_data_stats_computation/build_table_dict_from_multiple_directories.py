import sys
sys.path.append(".")
sys.path.append("../../")
from Code._csv_preprocessing import *
from prepare_data import *

def build_table_dict_from_multiple_directories(directories: list[str], outpath: str, table_set: set[str]) -> dict[str:pd.DataFrame]:
    out = {}
    for dir in directories:
        out.update(generate_table_dict(dir))
    with open(outpath, 'wb') as f:
        pickle.dump(out, f)
    return out

if __name__ == '__main__':
    dir_list_gittables = ['/home/francesco.pugnaloni/armadillo_all/datasets/gittables/csv/train_csv/', 
                          '/home/francesco.pugnaloni/armadillo_all/datasets/gittables/csv/test_csv/', 
                          '/home/francesco.pugnaloni/armadillo_all/datasets/gittables/csv/valid_csv/']
    root_gittables = '/home/francesco.pugnaloni/armadillo_all/datasets/gittables/'
    train_gittables = pd.read_csv(root_gittables+'/train.csv')
    test_gittables = pd.read_csv(root_gittables+'/test.csv')
    valid_gittables = pd.read_csv(root_gittables+'/valid.csv')
    # gittables_table_set = get_table_set([train_gittables, test_gittables, valid_gittables])
    outpath_table_dict_gittables = '/home/francesco.pugnaloni/armadillo_all/datasets/gittables/dictionaries/table_dictionaries/table_dict.pkl'


    dir_list_wikilast = ['/home/francesco.pugnaloni/armadillo_all/datasets/wikilast/csv/']
    root_wikilast = '/home/francesco.pugnaloni/armadillo_all/datasets/wikilast/'
    train_wikilast = pd.read_csv(root_wikilast+'/train.csv')
    test_wikilast = pd.read_csv(root_wikilast+'/test.csv')
    valid_wikilast = pd.read_csv(root_wikilast+'/valid.csv')
    # wikilast_table_set = get_table_set([train_wikilast, test_wikilast, valid_wikilast])
    wikilast_test_table_set = get_table_set([test_wikilast])
    outpath_table_dict_wikilast = '/home/francesco.pugnaloni/armadillo_all/datasets/wikilast/dictionaries/table_dictionaries/table_dict.pkl'
    outpath_table_dict_wikilast_test = '/home/francesco.pugnaloni/armadillo_all/datasets/wikilast/dictionaries/table_dictionaries/table_dict_test.pkl'
    #build_table_dict_from_multiple_directories(dir_list_gittables, outpath_table_dict_gittables, table_set=gittables_table_set)
    build_table_dict_from_multiple_directories(dir_list_wikilast, outpath_table_dict_wikilast_test, table_set=wikilast_test_table_set)
