import pickle
from tqdm import tqdm
def merge_dicts(ld: list) -> dict:
    out = {}
    for d in tqdm(ld):
        with open(d,'rb') as f:
            out.update(pickle.load(f))
    return out

if __name__ == '__main__':
    emb_dir = '/home/francesco.pugnaloni/armadillo_all/datasets/WikiTables/dictionaries/embedding_dictionaries/embdi/'
    out_dir = '/home/francesco.pugnaloni/armadillo_all/datasets/WikiTables/dictionaries/embedding_dictionaries/'
    embeddings_dicts = [emb_dir+'embedding_dict_train_0.pkl',emb_dir+'embedding_dict_train_1.pkl',emb_dir+'embedding_dict_train_2.pkl',emb_dir+'embedding_dict_train_3.pkl',
                        emb_dir+'embedding_dict_train_4.pkl',emb_dir+'embedding_dict_train_5.pkl',emb_dir+'embedding_dict_train_6.pkl',emb_dir+'embedding_dict_train_7.pkl',
                        emb_dir+'embedding_dict_train_8.pkl',emb_dir+'embedding_dict_train_9.pkl',
                        emb_dir+'embedding_dict_test.pkl', emb_dir+'embedding_dict_valid.pkl'
                        ]
    t_exec_dicts = [emb_dir+'t_execs_train_0.pkl',emb_dir+'t_execs_train_1.pkl',emb_dir+'t_execs_train_2.pkl',emb_dir+'t_execs_train_3.pkl',emb_dir+'t_execs_train_4.pkl',
                    emb_dir+'t_execs_train_5.pkl',emb_dir+'t_execs_train_6.pkl',emb_dir+'t_execs_train_7.pkl',emb_dir+'t_execs_train_8.pkl',emb_dir+'t_execs_train_9.pkl',
                    emb_dir+'t_execs_test.pkl',emb_dir+'t_execs_valid.pkl'
                    ]
    emb = merge_dicts(embeddings_dicts)
    time = merge_dicts(t_exec_dicts)
    with open(out_dir+'emb_dict_embdi.pkl', 'wb') as f:
        pickle.dump(emb,f)
    with open(out_dir+'t_execs_embdi.pkl', 'wb') as f:
        pickle.dump(time,f)