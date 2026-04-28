import sys
sys.path.append(".")
sys.path.append("./sloth/")
sys.path.append("../../")
from Baselines.sloth.sloth.sloth import sloth
import pandas as pd
import pickle
from tqdm import tqdm

def from_df_to_column_list(df: pd.DataFrame) -> list[list]:
    out = []
    for c in range(df.shape[1]):
        out.append([str(i) for i in df.iloc[:,c]])
        # out.append(list(df.iloc[:,c]))
    return out

def compute_overlap(t1: pd.DataFrame | str, t2: pd.DataFrame | str, verbose: bool=False) -> set[float, int]:
    if isinstance(t1, str):
        t1 = pd.read_csv(t1)
    if isinstance(t2, str):
        t2 = pd.read_csv(t2)
    min_area = min(t1.shape[0]*t1.shape[1], t2.shape[0]*t2.shape[1])
    t1 = from_df_to_column_list(t1)
    t2 = from_df_to_column_list(t2)
    out = sloth(t1, t2, metrics=[], verbose=verbose)
    try:
        return out / min_area, out
    except:
        return -1, -1

def re_avaluate_values(table_dict: str | dict, triple_dataset: str | pd.DataFrame, out_path: str=None, error_count: int=60_000) -> list:
    wrong_labels = []
    wrong_tables_data = {
        'r_id':[],
        'r_rows':[],
        'r_cols':[],
        'r_area':[],
        's_id':[],
        's_rows':[],
        's_cols':[],
        's_area':[],
        'a_overlap_recomputed':[],
        'a%_recomputed':[],
        'a%_test_dataset':[]
    }
    if isinstance(table_dict, str):
        with open(table_dict, 'rb') as f:
            table_dict = pickle.load(f)
    if isinstance(triple_dataset, str):
        triple_dataset = pd.read_csv(triple_dataset)
    for r in tqdm(range(triple_dataset.shape[0])):
        t1 = table_dict[str(triple_dataset.iloc[r].loc['r_id'])]
        t2 = table_dict[str(triple_dataset.iloc[r].loc['s_id'])]
        try:
            overlap_expected = triple_dataset.iloc[r].loc['a%']
        except:
            overlap_expected = triple_dataset.iloc[r].loc['a%_test_dataset']
        overlap_exact, a_overlap_exact = compute_overlap(t1, t2)
        if (abs(overlap_expected-overlap_exact) > 0.001) and (overlap_exact != -1):
            wrong_labels.append(r)
            print(f'Wrong label at row {r}, exact {overlap_exact}, expected {overlap_expected}')
            print(f'Progress: {len(wrong_labels)}/{error_count}')
            wrong_tables_data['r_id'].append(str(triple_dataset.iloc[r].loc['r_id']))
            wrong_tables_data['r_rows'].append(t1.shape[0])
            wrong_tables_data['r_cols'].append(t1.shape[1])
            wrong_tables_data['r_area'].append(t1.shape[0]*t1.shape[1])
            wrong_tables_data['s_id'].append(str(triple_dataset.iloc[r].loc['s_id']))
            wrong_tables_data['s_rows'].append(t2.shape[0])
            wrong_tables_data['s_cols'].append(t2.shape[1])
            wrong_tables_data['s_area'].append(t2.shape[0]*t2.shape[1])
            wrong_tables_data['a_overlap_recomputed'].append(a_overlap_exact)
            wrong_tables_data['a%_recomputed'].append(overlap_exact)
            wrong_tables_data['a%_test_dataset'].append(overlap_expected)
            if len(wrong_labels) >= error_count:
                break

    if out_path != None:
        # triple_dataset.iloc[wrong_labels,:].to_csv(out_path, index=False)
        pd.DataFrame(wrong_tables_data).to_csv(out_path, index=False)
    print(f'Number of wrong labels: {len(wrong_labels)}')
    print(pd.DataFrame(wrong_tables_data))
    return triple_dataset.iloc[wrong_labels,:]

if __name__ == '__main__':
    # print('Starting git')
    # re_avaluate_values(table_dict='/home/francesco.pugnaloni/armadillo_all/datasets/gittables/dictionaries/table_dictionaries/table_dict.pkl',
    #                    triple_dataset='/home/francesco.pugnaloni/armadillo_all/datasets/gittables/test.csv',
    #                 #    triple_dataset='/home/francesco.pugnaloni/armadillo_all/datasets/gittables/tmp/wrong_labels_test.csv',
    #                 #    out_path=None
    #                    out_path='/home/francesco.pugnaloni/armadillo_all/datasets/gittables/tmp/wrong_labels_test.csv'
    #                    )

    # print('Starting wiki')
    # re_avaluate_values(table_dict='/home/francesco.pugnaloni/armadillo_all/datasets/wikilast/dictionaries/table_dictionaries/table_dict.pkl',
    #                    triple_dataset='/home/francesco.pugnaloni/armadillo_all/datasets/wikilast/test.csv',
    #                 #    triple_dataset='/home/francesco.pugnaloni/armadillo_all/datasets/wikilast/tmp/wrong_labels_test.csv',
    #                 #    out_path=None
    #                    out_path='/home/francesco.pugnaloni/armadillo_all/datasets/wikilast/tmp/wrong_labels_test.csv'
    #                    )

    # with open('/home/francesco.pugnaloni/armadillo_all/datasets/wikilast/dictionaries/table_dictionaries/table_dict.pkl', 'rb') as f:
    #     table_dict = pickle.load(f)
    # t1 = pd.read_csv('/home/francesco.pugnaloni/armadillo_all/datasets/wikilast/csv/123.66036.csv')
    # t2 = pd.read_csv('/home/francesco.pugnaloni/armadillo_all/datasets/wikilast/csv/123.66050.csv')
    # ov, ov_a = compute_overlap(t1=t1,t2=t2, verbose=True)
    # print(f'a%={ov}, ov_area={ov_a}')

    # t1_d = table_dict['123.66036.csv']
    # t2_d = table_dict['123.66050.csv']
    # ov_d, ov_a_d = compute_overlap(t1=t1_d,t2=t2_d, verbose=True)
    # print(f'a%={ov}, ov_area={ov_a}')
    # print('end')

    print('Starting wiki')

    re_avaluate_values(table_dict='/home/francesco.pugnaloni/armadillo_all/datasets/WikiTables/dictionaries/table_dict.pkl',
        #table_dict='/home/francesco.pugnaloni/armadillo_all/datasets/wikilast/dictionaries/table_dictionaries/table_dict.pkl',
                       triple_dataset='/home/francesco.pugnaloni/armadillo_all/datasets/old/wikilast/test.csv',
                    #    triple_dataset='/home/francesco.pugnaloni/armadillo_all/datasets/old/wikilast/tmp/wrong_labels_test.csv',
                    #    triple_dataset='/home/francesco.pugnaloni/armadillo_all/datasets/old/wikilast/tmp/wrong_labels_test_reduced_loading_cast_str.csv',
                       out_path='/home/francesco.pugnaloni/armadillo_all/datasets/tmp/wrong_labels.csv'
                    #    out_path='/home/francesco.pugnaloni/armadillo_all/datasets/wikilast/tmp/wrong_labels_test_reduced_loading_cast_str.csv'
                       )
