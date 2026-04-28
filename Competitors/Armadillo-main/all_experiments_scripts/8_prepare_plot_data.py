import sys
sys.path.append(".")
sys.path.append("../../")
import pickle
import pandas as pd
from tqdm import tqdm
from Baselines.embedding_scaling_model import Embedding_scaler
import time
import torch.nn.functional as F

class Get_time_per_pair:
    def __init__(self, mode: str, t_exec_dict: dict|str) -> None:
        self.mode = mode
        if isinstance(t_exec_dict, str):
            with open(t_exec_dict, 'rb') as f:
                t_exec_dict = pickle.load(f)
        self.t_exec_dict = t_exec_dict
    
    def get_emb_time(self, r_id, s_id) -> float:
        if self.mode == 'embdi':
            return self.t_exec_dict[f'{r_id}|{s_id}']
        elif self.mode=='armadillo':
            return self.t_exec_dict[r_id]['t_tot'] + self.t_exec_dict[s_id]['t_tot']
        else:
            return self.t_exec_dict[r_id] + self.t_exec_dict[s_id]
    
def add_new_columns(df: pd.DataFrame|str, t_exec_dict: dict|str, emb_dict: dict|str, model_checkpoint: str, model_name: str='std', correct_predictions: str=None) -> dict:
    if isinstance(df, str):
        df = pd.read_csv(df)
    if isinstance(t_exec_dict, str):
        with open(t_exec_dict, 'rb') as f:
            t_exec_dict = pickle.load(f)
    if isinstance(emb_dict, str):
        with open(emb_dict, 'rb') as f:
            emb_dict = pickle.load(f)
    if model_name == 'armadillo_wikilast' or model_name=='armadillo_gittables' or model_name=='armadillo_wikitables' or model_name=='cosine_similarity' or model_name=='armadillo_w_w' or model_name=='armadillo_g_w' or model_name=='armadillo_g_g' or model_name=='armadillo_w_g':
        pass
    else:
        model = Embedding_scaler.load_from_checkpoint(model_checkpoint)
    time_col_name = f'{model_name}_overlap_time'
    overlap_comp_times_list = []
    overlap_pred_col_name = f'{model_name}_overlap_pred'
    overlap_pred_list = []

    if model_name == 'embdi_wikilast' or model_name == 'embdi_gittables' or model_name == 'embdi_wikitables':
        mode = 'embdi'
    elif model_name == 'armadillo_wikilast' or model_name=='armadillo_gittables' or model_name=='armadillo_wikitables' or model_name=='cosine_similarity' or model_name=='armadillo_w_w' or model_name=='armadillo_g_w' or model_name=='armadillo_g_g' or model_name=='armadillo_w_g':
        mode = 'armadillo'
    else:
        mode= None
    embedding_times = Get_time_per_pair(mode=mode, t_exec_dict=t_exec_dict)
    
    for r in tqdm(range(df.shape[0])):
        r_id = df.iloc[r].loc['r_id']
        s_id = df.iloc[r].loc['s_id']
        emb_gen_time = embedding_times.get_emb_time(r_id, s_id)
        start_emb = time.time()
        if mode == 'embdi':
            r_emb, s_emb = emb_dict[f'{r_id}|{s_id}']
            r_emb = model.forward(r_emb.unsqueeze(dim=0))
            s_emb = model.forward(s_emb.unsqueeze(dim=0))
        elif mode == 'armadillo':
            r_emb = emb_dict[r_id]
            s_emb = emb_dict[s_id]
        else:
            r_emb = model.forward(emb_dict[r_id])
            s_emb = model.forward(emb_dict[s_id])
        try:
            cos_sim_start = time.time()
            overlap_pred_list.append(max(0, float(F.cosine_similarity(r_emb, s_emb))))
            cos_sim_end = time.time()
        except:
            overlap_pred_list.append(max(0, float(F.cosine_similarity(r_emb.unsqueeze(dim=0), s_emb.unsqueeze(dim=0)))))
        end_emb = time.time()
        if model_name=='cosine_similarity':
            overlap_time = cos_sim_end-cos_sim_start
        else:
            overlap_time = emb_gen_time + (end_emb-start_emb)
        overlap_comp_times_list.append(overlap_time)
    df[time_col_name] = overlap_comp_times_list
    if model_name != 'cosine_similarity':
        df[overlap_pred_col_name] = overlap_pred_list
        df[f'{model_name}_ae'] = abs(df['a%']-df[overlap_pred_col_name])
    if correct_predictions != None:
        with open(correct_predictions,'rb') as f:
            re_comp_arm = pickle.load(f)
        new_overlaps = re_comp_arm['predictions']
        df[overlap_pred_col_name]  = [float(n) for n in new_overlaps]
        df[f'{model_name}_ae'] = abs(df['a%']-df[overlap_pred_col_name])
    return df

if __name__ == '__main__':
    root = ''
    root_git = root+'/GitTables/'
    root_wiki = root+'/WikiTables/'
    df_wiki = pd.read_csv(root+'/WikiTables/evaluation/eval_wiki.csv')
    df_git = pd.read_csv(root+'/GitTables/evaluation/eval_wiki.csv')
    df_querying = pd.read_csv(root+'/GitTables/table_querying/evaluation/table_querying_results.csv')

    params_armadillo_wikilast_wikilast = {
        'df' : df_wiki, 
        't_exec_dict' : root+'/WikiTables/charts/embedding_gen_time_wikilast.pkl',
        'emb_dict' : root+'/WikiTables/charts/embedding_file_wiki_on_wiki.pkl',
        'model_checkpoint' : root+'/WikiTables/models/armadillo_wiki.pth',
        'model_name' : 'armadillo_w_w',
        'correct_predictions':root+'/WikiTables/evaluation/re_eval_armadillo_w_w.pkl'
    }


    params_armadillo_gittables_wikilast = {
        'df' : df_wiki, 
        't_exec_dict' : root+'/WikiTables/dictionaries/embedding_dictionaries/t_execs_armadillo_g_w.pkl',
        'emb_dict' : root+'/WikiTables/dictionaries/embedding_dictionaries/embedding_dict_test_set_armadillo_g_w.pkl',
        'model_checkpoint' : root+'/GitTables/models/armadillo_git.pth',
        'model_name' : 'armadillo_g_w',
        'correct_predictions':root+'/WikiTables/evaluation/re_eval_armadillo_g_w.pkl'
    }

    params_roberta_tables_gittables_wikilast = {
        'df' : df_wiki, 
        't_exec_dict' : root+'/WikiTables/dictionaries/embedding_dictionaries/t_execs_roberta_tables_300_300_wikilast.pkl',
        'emb_dict' : root+'/WikiTables/dictionaries/embedding_dictionaries/emb_dict_roberta_tables_300_300.pkl',
        'model_checkpoint' : root+'/GitTables/models/roberta_tables_300_300_gittables.pth',
        'model_name' : 'roberta_tables_300_300_gittables' 
    }


    params_roberta_tables_300_300_wikilast = {
        'df' : df_wiki, 
        't_exec_dict' : root+'/WikiTables/dictionaries/embedding_dictionaries/t_execs_roberta_tables_300_300_wikilast.pkl',
        'emb_dict' : root+'/WikiTables/dictionaries/embedding_dictionaries/emb_dict_roberta_tables_300_300.pkl',
        'model_checkpoint' : root+'/WikiTables/models/roberta_tables_300_300_wikilast.pth',
        'model_name' : 'roberta_tables_300_300_wikilast'
    }

    params_cosine_similarity = {
        'df' : df_wiki, 
        't_exec_dict' : root+'/WikiTables/charts/embedding_gen_time_wikilast.pkl',
        'emb_dict' : root+'/WikiTables/charts/embedding_file_wiki_on_wiki.pkl',
        'model_checkpoint' : root+'/WikiTables/models/armadillo_wiki.pth',
        'model_name' : 'cosine_similarity'
    }
    params_embdi_wikilast = {
        'df' : df_wiki, 
        't_exec_dict' : root+'/WikiTables/dictionaries/embedding_dictionaries/t_execs_embdi.pkl',
        'emb_dict' : root+'/WikiTables/dictionaries/embedding_dictionaries/emb_dict_embdi.pkl',
        'model_checkpoint' : root+'/WikiTables/models/embdi_wikilast.pth',
        'model_name' : 'embdi_wikilast'
    }

    params_turl_tables_wikilast = {
        'df' : df_wiki, 
        't_exec_dict' : root+'/WikiTables/dictionaries/embedding_dictionaries/t_execs_turl_tables_300_300_wikilast.pkl',
        'emb_dict' : root+'/WikiTables/dictionaries/embedding_dictionaries/emb_dict_turl_tables_300_300.pkl',
        'model_checkpoint' : root+'/WikiTables/models/turl_tables_300_300_wikilast.pth',
        'model_name' : 'turl_wikilast'
    }

    params_bert_tables_300_300_wikilast = {
        'df' : df_wiki, 
        't_exec_dict' : root+'/WikiTables/dictionaries/embedding_dictionaries/t_execs_bert_tables_300_300_wikilast.pkl',
        'emb_dict' : root+'/WikiTables/dictionaries/embedding_dictionaries/emb_dict_bert_tables_300_300.pkl',
        'model_checkpoint' : root+'/WikiTables/models/bert_tables_300_300_wikilast.pth',
        'model_name' : 'bert_tables_300_300_wikilast'
    }

    params_bert_tables_300_300_wikilast_anon = {
        'df' : df_wiki, 
        't_exec_dict' : root+'/WikiTables/dictionaries/embedding_dictionaries/t_execs_bert_tables_anon_300_300_wikilast.pkl',
        'emb_dict' : root+'/WikiTables/dictionaries/embedding_dictionaries/emb_dict_bert_tables_anon_300_300.pkl',
        'model_checkpoint' : root+'/WikiTables/models/bert_tables_anon_300_300.pth',
        'model_name' : 'bert_tables_anon_300_300_wikilast'
    }

    params_roberta_tables_300_300_wikilast_anon = {
        'df' : df_wiki, 
        't_exec_dict' : root+'/WikiTables/dictionaries/embedding_dictionaries/t_execs_roberta_tables_anon_300_300_wikilast.pkl',
        'emb_dict' : root+'/WikiTables/dictionaries/embedding_dictionaries/emb_dict_roberta_tables_anon_300_300.pkl',
        'model_checkpoint' : root+'/WikiTables/models/roberta_tables_anon_300_300.pth',
        'model_name' : 'roberta_tables_anon_300_300_wikilast'
    }

    params_bert_rows_300_300_wikilast = {
        'df' : df_wiki, 
        't_exec_dict' : root+'/WikiTables/dictionaries/embedding_dictionaries/t_execs_bert_rows_300_300_wikilast.pkl',
        'emb_dict' : root+'/WikiTables/dictionaries/embedding_dictionaries/emb_dict_bert_lines_300_300.pkl',
        'model_checkpoint' : root+'/WikiTables/models/bert_rows_300_300_wikilast.pth',
        'model_name' : 'bert_rows_300_300_wikilast'
    }

    params_roberta_rows_300_300_wikilast = {
        'df' : df_wiki, 
        't_exec_dict' : root+'/WikiTables/dictionaries/embedding_dictionaries/t_execs_roberta_rows_300_300_wikilast.pkl',
        'emb_dict' : root+'/WikiTables/dictionaries/embedding_dictionaries/emb_dict_roberta_lines_300_300.pkl',
        'model_checkpoint' : root+'/WikiTables/models/roberta_rows_300_300_wikilast.pth',
        'model_name' : 'roberta_rows_300_300_wikilast'
    }

    params_wiki = [
        params_armadillo_wikilast_wikilast,
        params_armadillo_gittables_wikilast,
        params_roberta_tables_gittables_wikilast,
        params_cosine_similarity,
        params_embdi_wikilast,
        params_turl_tables_wikilast,
        params_bert_tables_300_300_wikilast,
        params_bert_tables_300_300_wikilast_anon,
        params_roberta_tables_300_300_wikilast,
        params_roberta_tables_300_300_wikilast_anon,
        params_bert_rows_300_300_wikilast,
        params_roberta_rows_300_300_wikilast
    ]
    for p in params_wiki:
        print(f"Processing {p['model_name']}")
        df_wiki = add_new_columns(**p)
    
    df_wiki.to_csv(root+'/WikiTables/evaluation/eval_wiki.csv', index=False)

    params_armadillo_gittables_gittables = {
        'df' : df_git, 
        't_exec_dict' : root+'/GitTables/dictionaries/embedding_dictionaries/t_execs_armadillo_g_g.pkl',
        'emb_dict' : root+'/GitTables/dictionaries/embedding_dictionaries/embedding_dict_test_set_armadillo_g_g.pkl',
        'model_checkpoint' : root+'/GitTables/models/armadillo_git.pth',
        'model_name' : 'armadillo_g_g',
        'correct_predictions':root+'/GitTables/evaluation/re_eval_armadillo_g_g.pkl'
    }
    params_armadillo_wikilast_gittables = {
        'df' : df_git, 
        't_exec_dict' : root+'/GitTables/dictionaries/embedding_dictionaries/t_execs_armadillo_w_g.pkl',
        'emb_dict' : root+'/GitTables/dictionaries/embedding_dictionaries/embedding_dict_test_set_armadillo_w_g.pkl',
        'model_checkpoint' : root+'/WikiTables/models/armadillo_wiki.pth',
        'model_name' : 'armadillo_w_g',
        'correct_predictions':root+'/GitTables/evaluation/re_eval_armadillo_w_g.pkl'
    }
    params_roberta_tables_git_git = {
        'df' : df_git, 
        't_exec_dict' : root+'/GitTables/dictionaries/embedding_dictionaries/t_execs_roberta_tables_300_300_gittables.pkl',
        'emb_dict' : root+'/GitTables/dictionaries/embedding_dictionaries/emb_dict_roberta_tables_300_300.pkl',
        'model_checkpoint' : root+'/GitTables/models/roberta_tables_300_300_gittables.pth',
        'model_name' : 'roberta_t_g_g'
    }
    params_roberta_tables_wiki_git = {
        'df' : df_git, 
        't_exec_dict' : root+'/GitTables/dictionaries/embedding_dictionaries/t_execs_roberta_tables_300_300_gittables.pkl',
        'emb_dict' : root+'/GitTables/dictionaries/embedding_dictionaries/emb_dict_roberta_tables_300_300.pkl',
        'model_checkpoint':root+'/WikiTables/models/roberta_tables_300_300_wikilast.pth',
        'model_name':'roberta_t_w_g'
    }
    params_cosine_similarity_git = {
        'df' : df_git, 
        't_exec_dict' : root+'/GitTables/dictionaries/embedding_dictionaries/t_execs_armadillo_g_g.pkl',
        'emb_dict' : root+'/GitTables/dictionaries/embedding_dictionaries/embedding_dict_test_set_armadillo_g_g.pkl',
        'model_checkpoint' : root+'/GitTables/models/armadillo_git.pth',
        'model_name' : 'cosine_similarity'
    }
    params_git = [
        params_armadillo_gittables_gittables,
        params_armadillo_wikilast_gittables,
        params_roberta_tables_git_git,
        params_roberta_tables_wiki_git,
        params_cosine_similarity_git
    ]

    for p in params_git:
        print(f"Processing {p['model_name']}")
        df_git = add_new_columns(**p)
    df_git.to_csv(root+'/GitTables/evaluation/eval_wiki.csv', index=False)

    params_bert_r = {
        'df':df_querying,
        't_exec_dict':root+'/GitTables/table_querying/dictionaries/embedding_dictionaries/t_execs_bert_r.pkl',
        'emb_dict':root+'/GitTables/table_querying/dictionaries/embedding_dictionaries/emb_dict_bert_r.pkl',
        'model_checkpoint':root+'/GitTables/models/bert_rows_300_300_gittables.pth',
        'model_name':'bert_r'
    }
    params_bert_t = {
        'df':df_querying,
        't_exec_dict':root+'/GitTables/table_querying/dictionaries/embedding_dictionaries/t_execs_bert_t.pkl',
        'emb_dict':root+'/GitTables/table_querying/dictionaries/embedding_dictionaries/emb_dict_bert_t.pkl',
        'model_checkpoint':root+'/GitTables/models/bert_tables_300_300_gittables.pth',
        'model_name':'bert_t'
    }
    params_roberta_r = {
        'df':df_querying,
        't_exec_dict':root+'/GitTables/table_querying/dictionaries/embedding_dictionaries/t_execs_roberta_r.pkl',
        'emb_dict':root+'/GitTables/table_querying/dictionaries/embedding_dictionaries/emb_dict_roberta_r.pkl',
        'model_checkpoint':root+'/GitTables/models/roberta_rows_300_300_gittables.pth',
        'model_name':'roberta_r'
    }
    params_roberta_t = {
        'df':df_querying,
        't_exec_dict':root+'/GitTables/table_querying/dictionaries/embedding_dictionaries/t_execs_roberta_t.pkl',
        'emb_dict':root+'/GitTables/table_querying/dictionaries/embedding_dictionaries/emb_dict_roberta_t.pkl',
        'model_checkpoint':root+'/GitTables/models/roberta_tables_300_300_gittables.pth',
        'model_name':'roberta_t'
    }
    param_turl = {
        'df':df_querying,
        't_exec_dict':root+'/GitTables/table_querying/dictionaries/embedding_dictionaries/full_t_execs_dict_turl.pkl',
        'emb_dict':root+'/GitTables/table_querying/dictionaries/embedding_dictionaries/full_emb_dict_turl.pkl',
        'model_checkpoint':root+'/GitTables/models/turl_tables_300_300_gittables.pth',
        'model_name':'turl'
    }
    params_armadillo = {
        'df':df_querying,
        't_exec_dict':root+'/GitTables/table_querying/dictionaries/embedding_dictionaries/t_execs_armadillo_querying.pkl',
        'emb_dict':root+'/GitTables/table_querying/dictionaries/embedding_dictionaries/embedding_armadillo_querying.pkl',
        'model_checkpoint':root+'/GitTables/models/armadillo_git.pth',
        'model_name':'armadillo_g_g',
        'correct_predictions':root+'/GitTables/table_querying/dictionaries/embedding_dictionaries/re_eval_table_querying_armadillo_g.pkl'
    }
    params_querying = [
        params_armadillo,
        param_turl,
        params_bert_r,
        params_bert_t,
        params_roberta_r,
        params_roberta_t
    ]
    print('Running table querying experiment')
    for p in params_querying:
        print(f"Processing {p['model_name']}")
        df_querying = add_new_columns(**p)
    df_querying.to_csv(root+'/GitTables/table_querying/evaluation/table_querying_results_with_turl_to_merge.csv', index=False)