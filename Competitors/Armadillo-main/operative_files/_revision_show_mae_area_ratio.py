import pandas as pd
import pickle
import matplotlib.pyplot as plt
import pickle
import pandas as pd
from tqdm import tqdm
import numpy as np
import seaborn as sns

def add_area_area_ratio(df: str|pd.DataFrame, table_dict: str|dict, outpath: str=None) -> pd.DataFrame:
    if isinstance(df, str):
        df = pd.read_csv(df)
    if isinstance(table_dict, str):
        with open(table_dict, 'rb') as f:
            table_dict = pickle.load(f)
    ratios = {'area_ratio': []}
    for r in tqdm(range(df.shape[0])):
        r_id = df.iloc[r]['r_id']
        s_id = df.iloc[r]['s_id']

        r_area = table_dict[r_id].shape[0]*table_dict[r_id].shape[1]
        s_area = table_dict[s_id].shape[0]*table_dict[s_id].shape[1]

        area_ratio = min(r_area, s_area)/max(r_area, s_area)

        ratios['area_ratio'].append(area_ratio)
    
    df['area_ratio'] = ratios['area_ratio']
    if outpath is not None:
        df.to_csv(outpath, index=False)
    return df


def compare_models_hist(data: pd.DataFrame | str, approaches: dict, bin_criterion: str='area_ratio', bins_name: str='Area ratios', out_pdf: str=None, font_scale: float=1.45, n_col: int=4, fig_size=(8,6)) -> None:
    if isinstance(data, str):
        data = pd.read_csv(data)
    ranges = f'{bins_name} Range' 
    new_data = {
        ranges:[],
        'Approach':[],
        'MAE':[]
    }
    for i in range(1, 11, 1):
        i /= 10
        prev = round(i-0.1, 2)
        t = data[data[bin_criterion] >= prev]
        if i == 1:
            t = t[t[bin_criterion] <= i]
        else:
            t = t[t[bin_criterion] < i] 
        curr =  f'[{prev},\n{i}]'
        for k in approaches.keys():
            new_data['Approach'].append(approaches[k])
            new_data[ranges].append(curr)
            new_data['MAE'].append(round(np.mean(t[k]),2))
    plt.figure(figsize=fig_size)
    df = pd.DataFrame(new_data)
    sns.set_theme(font_scale=font_scale, style="whitegrid")
    ax = sns.barplot(data=df, x=ranges, y='MAE', hue='Approach')
    plt.grid(False)
    plt.tick_params(left = True) 
    plt.legend(bbox_to_anchor=(0.5, 1.29), loc='upper center', ncol=n_col)
    if isinstance(out_pdf, str):
        plt.tight_layout()
        plt.savefig(out_pdf, format="pdf", bbox_inches='tight')

if __name__ == '__main__':
    # GitTables    
    df = add_area_area_ratio(df='/home/francesco.pugnaloni/armadillo_all/datasets/GitTables/evaluation/eval_wiki.csv', table_dict='/home/francesco.pugnaloni/armadillo_all/datasets/GitTables/dictionaries/table_dictionaries/table_dict.pkl', outpath='/home/francesco.pugnaloni/armadillo_all/datasets/GitTables/evaluation/eval_git_with_area_ratios.csv')

    # WikiTables
    df = add_area_area_ratio(df='/home/francesco.pugnaloni/armadillo_all/datasets/WikiTables/evaluation/eval_wiki.csv', table_dict='/home/francesco.pugnaloni/armadillo_all/datasets/WikiTables/dictionaries/table_dict.pkl', outpath='/home/francesco.pugnaloni/armadillo_all/datasets/WikiTables/evaluation/eval_wiki_with_area_ratios.csv')