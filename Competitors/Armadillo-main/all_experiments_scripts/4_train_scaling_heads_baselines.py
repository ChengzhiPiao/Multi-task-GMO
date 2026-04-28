import sys
sys.path.append(".")
sys.path.append("../../")
from Baselines.embedding_scaling_model import *
import pickle
import os
import time
from tqdm import tqdm

def build_dataloader(input: str | pd.DataFrame, embedding_dictionary:  str | dict[str, torch.Tensor], batch_size: int=2, is_embdi: bool=False, remove_missing_pairs: bool=False) -> pd.DataFrame:
    if isinstance(input, str):
        input = pd.read_csv(input)
    if isinstance(embedding_dictionary, str):
        with open(embedding_dictionary, 'rb') as f:
            embedding_dictionary = pickle.load(f)
    to_drop = []
    for r in tqdm(range(input.shape[0])):
        if (embedding_dictionary[input.iloc[r].loc['r_id']] == None) or (embedding_dictionary[input.iloc[r].loc['s_id']] == None):
            to_drop.append(r)
    print(f'Dropping {len(to_drop)}/{input.shape[0]} samples')
    input = input.drop(index=to_drop)

    if is_embdi:
        return DataLoader(Embedding_dataset_embdi(input, embedding_dictionary), num_workers=0, batch_size=batch_size)
    else:
        return DataLoader(Embedding_dataset(input, embedding_dictionary), num_workers=0, batch_size=batch_size)

def train_scaling_model(train: pd.DataFrame | str, test: pd.DataFrame | str, valid: pd.DataFrame | str, input_size: int, lr: float,
                        embedding_dictionary: dict[str, torch.Tensor] | str, trainer_config: dict, model_checkpoint_dir: str=None, model_name: str=None,
                        out_path: str=None, batch_size: int=2, patience: int=5, is_embdi: bool=False, remove_missing_pairs: bool=False) -> Embedding_scaler:
    
    start = time.time()
    if isinstance(embedding_dictionary, str):
        with open(embedding_dictionary, 'rb') as f:
            embedding_dictionary = pickle.load(f)
    out_path = out_path+model_name+'.pth'
    model_checkpoint_dir = model_checkpoint_dir+model_name
    if not os.path.exists(model_checkpoint_dir):
        os.makedirs(model_checkpoint_dir)

    train = build_dataloader(train, embedding_dictionary, batch_size=batch_size, is_embdi=is_embdi, remove_missing_pairs=remove_missing_pairs)
    test = build_dataloader(test, embedding_dictionary, batch_size=batch_size, is_embdi=is_embdi, remove_missing_pairs=remove_missing_pairs)
    valid = build_dataloader(valid, embedding_dictionary, batch_size=batch_size, is_embdi=is_embdi, remove_missing_pairs=remove_missing_pairs)

    model = Embedding_scaler(input_size=input_size, output_size=300, lr=lr)
    callbacks = [
        ModelCheckpoint(dirpath=model_checkpoint_dir, filename=model_name),
        EarlyStopping(monitor='val_loss', patience=patience, mode='min')
        ]
    trainer = L.Trainer(**trainer_config, callbacks=callbacks)
    trainer.fit(model, train_dataloaders=train, val_dataloaders=valid)
    trainer.test(model, dataloaders=test)
    # torch.save(model.state_dict(), out_path)
    trainer.save_checkpoint(out_path)
    end = time.time()
    print(f'Total training time: {end-start}s')
    return model
root = ''
root_git = root+'/GitTables/'
root_wiki = root+'/WikiTables/'

checkpoints_git= root_git+'/models/checkpoints/'

checkpoints_wiki = root_wiki+'/models/checkpoints/'

if __name__ == '__main__':
    if not os.path.exists(root_git+'/models/checkpoints/'):
        os.makedirs(root_git+'/models/checkpoints/')
    if not os.path.exists(root_wiki+'/models/checkpoints/'):
        os.makedirs(root_wiki+'/models/checkpoints/')
        #_____________________________ROWS GRANULARITY
    params_bert_rows_300_300_gittables={
        'train':root_git+'train.csv',
        'test':root_git+'test.csv',
        'valid':root_git+'valid.csv',
        'embedding_dictionary':root_git+'dictionaries/embedding_dictionaries/'+'emb_dict_bert_lines_300_300.pkl',
        'out_path':root_git+'models/',
        'input_size':768,
        'model_name':'bert_rows_300_300_gittables',
        'model_checkpoint_dir':checkpoints_git
    }

    params_roberta_rows_300_300_gittables={
        'train':root_git+'train.csv',
        'test':root_git+'test.csv',
        'valid':root_git+'valid.csv',
        'embedding_dictionary':root_git+'dictionaries/embedding_dictionaries/'+'emb_dict_roberta_lines_300_300.pkl',
        'out_path':root_git+'models/',
        'input_size':768,
        'model_name':'roberta_rows_300_300_gittables',
        'model_checkpoint_dir':checkpoints_git
    }

    params_bert_rows_300_300_wikilast={
        'train':root_wiki+'train.csv',
        'test':root_wiki+'test.csv',
        'valid':root_wiki+'valid.csv',
        'embedding_dictionary':root_wiki+'dictionaries/embedding_dictionaries/'+'emb_dict_bert_lines_300_300.pkl',
        'out_path':root_wiki+'models/',
        'input_size':768,
        'model_name':'bert_rows_300_300_wikilast',
        'model_checkpoint_dir':checkpoints_wiki
    }

    params_roberta_rows_300_300_wikilast={
        'train':root_wiki+'train.csv',
        'test':root_wiki+'test.csv',
        'valid':root_wiki+'valid.csv',
        'embedding_dictionary':root_wiki+'dictionaries/embedding_dictionaries/'+'emb_dict_roberta_lines_300_300.pkl',
        'out_path':root_wiki+'models/',
        'input_size':768,
        'model_name':'roberta_rows_300_300_wikilast',
        'model_checkpoint_dir':checkpoints_wiki
    }

    #______________________________TABLE GRANULARITY
    params_bert_tables_300_300_gittables={
        'train':root_git+'train.csv',
        'test':root_git+'test.csv',
        'valid':root_git+'valid.csv',
        'embedding_dictionary':root_git+'dictionaries/embedding_dictionaries/'+'emb_dict_bert_tables_300_300.pkl',
        'out_path':root_git+'models/',
        'input_size':768,
        'model_name':'bert_tables_300_300_gittables',
        'model_checkpoint_dir':checkpoints_git
    }

    params_roberta_tables_300_300_gittables={
        'train':root_git+'train.csv',
        'test':root_git+'test.csv',
        'valid':root_git+'valid.csv',
        'embedding_dictionary':root_git+'dictionaries/embedding_dictionaries/'+'emb_dict_roberta_tables_300_300.pkl',
        'out_path':root_git+'models/',
        'input_size':768,
        'model_name':'roberta_tables_300_300_gittables',
        'model_checkpoint_dir':checkpoints_git
    }

    params_bert_tables_300_300_wikilast={
        'train':root_wiki+'train.csv',
        'test':root_wiki+'test.csv',
        'valid':root_wiki+'valid.csv',
        'embedding_dictionary':root_wiki+'dictionaries/embedding_dictionaries/'+'emb_dict_bert_tables_300_300.pkl',
        'out_path':root_wiki+'models/',
        'input_size':768,
        'model_name':'bert_tables_300_300_wikilast',
        'model_checkpoint_dir':checkpoints_wiki
    }

    params_roberta_tables_300_300_wikilast={
        'train':root_wiki+'train.csv',
        'test':root_wiki+'test.csv',
        'valid':root_wiki+'valid.csv',
        'embedding_dictionary':root_wiki+'dictionaries/embedding_dictionaries/'+'emb_dict_roberta_tables_300_300.pkl',
        'out_path':root_wiki+'models/',
        'input_size':768,
        'model_name':'roberta_tables_300_300_wikilast',
        'model_checkpoint_dir':checkpoints_wiki
    }

    #_______________________TURL

    params_turl_tables_300_300_wikilast={
        'train':root_wiki+'train.csv',
        'test':root_wiki+'test.csv',
        'valid':root_wiki+'valid.csv',
        'embedding_dictionary':root_wiki+'dictionaries/embedding_dictionaries/'+'emb_dict_turl_tables_300_300.pkl',
        'out_path':root_wiki+'models/',
        'input_size':312,
        'model_name':'turl_tables_300_300_wikilast',
        'model_checkpoint_dir':checkpoints_wiki
    }

    params_turl_tables_300_300_gittables={
        'train':root_git+'train.csv',
        'test':root_git+'test.csv',
        'valid':root_git+'valid.csv',
        'embedding_dictionary':root_git+'/dictionaries/embedding_dictionaries/emb_dict_turl_tables_128_128.pkl',
        'out_path':root_git+'models/',
        'input_size':312,
        'model_name':'turl_tables_300_300_gittables',
        'model_checkpoint_dir':checkpoints_git,
        'remove_missing_pairs':True
    }
    #_______________________embdi
    params_embdi_wiki={
        'train':root_wiki+'train.csv',
        'test':root_wiki+'test.csv',
        'valid':root_wiki+'valid.csv',
        'embedding_dictionary':root_wiki+'dictionaries/embedding_dictionaries/'+'emb_dict_embdi.pkl',
        'out_path':root_wiki+'models/',
        'input_size':300,
        'model_name':'embdi_wikilast',
        'model_checkpoint_dir':checkpoints_wiki,
        'is_embdi':True
    }

    #________________________anon_models
    params_roberta_tables_anon_300_300_wikilast={
        'train':root_wiki+'train.csv',
        'test':root_wiki+'test.csv',
        'valid':root_wiki+'valid.csv',
        'embedding_dictionary':root_wiki+'dictionaries/embedding_dictionaries/'+'emb_dict_roberta_tables_anon_300_300.pkl',
        'out_path':root_wiki+'models/',
        'input_size':768,
        'model_name':'roberta_tables_anon_300_300',
        'model_checkpoint_dir':checkpoints_wiki
    }

    params_bert_tables_anon_300_300_wikilast={
        'train':root_wiki+'train.csv',
        'test':root_wiki+'test.csv',
        'valid':root_wiki+'valid.csv',
        'embedding_dictionary':root_wiki+'dictionaries/embedding_dictionaries/'+'emb_dict_bert_tables_anon_300_300.pkl',
        'out_path':root_wiki+'models/',
        'input_size':768,
        'model_name':'bert_tables_anon_300_300',
        'model_checkpoint_dir':checkpoints_wiki
    }

    #________________________________________________________________________________________________________
    trainer_config =   {
        'accelerator': "gpu",
        'fast_dev_run' : False, 
        'deterministic' : True, 
        'devices':[0],
        'min_epochs':1,
        'max_epochs':100,
        'log_every_n_steps':50,
    }

    train_scaling_model(**params_bert_rows_300_300_wikilast, lr=0.01, trainer_config=trainer_config, batch_size=8_192, patience=10)
    train_scaling_model(**params_roberta_rows_300_300_wikilast, lr=0.01, trainer_config=trainer_config, batch_size=8_192, patience=10)

    train_scaling_model(**params_bert_tables_300_300_wikilast, lr=0.01, trainer_config=trainer_config, batch_size=8_192, patience=10)
    train_scaling_model(**params_roberta_tables_300_300_wikilast, lr=0.01, trainer_config=trainer_config, batch_size=8_192, patience=10)
    train_scaling_model(**params_turl_tables_300_300_wikilast, lr=0.01, trainer_config=trainer_config, batch_size=8_192, patience=10)

    train_scaling_model(**params_embdi_wiki, lr=0.01, trainer_config=trainer_config, batch_size=8_192, patience=10)
    train_scaling_model(**params_roberta_tables_anon_300_300_wikilast, lr=0.01, trainer_config=trainer_config, batch_size=8_192, patience=10)
    train_scaling_model(**params_bert_tables_anon_300_300_wikilast, lr=0.01, trainer_config=trainer_config, batch_size=8_192, patience=10)


    train_scaling_model(**params_bert_rows_300_300_gittables, lr=0.01, trainer_config=trainer_config, batch_size=8_192, patience=10)
    train_scaling_model(**params_roberta_rows_300_300_gittables, lr=0.01, trainer_config=trainer_config, batch_size=8_192, patience=10)

    train_scaling_model(**params_bert_tables_300_300_gittables, lr=0.01, trainer_config=trainer_config, batch_size=8_192, patience=10)
    train_scaling_model(**params_roberta_tables_300_300_gittables, lr=0.01, trainer_config=trainer_config, batch_size=8_192, patience=10)

    train_scaling_model(**params_turl_tables_300_300_gittables, lr=0.01, trainer_config=trainer_config, batch_size=8_192, patience=10)

    #____________________________________________________________________________________________________________________________________________________________________________________
    model = Embedding_scaler.load_from_checkpoint(root_wiki+'/models/bert_rows_300_300_wikilast.pth')
    trainer =  L.Trainer(**trainer_config, callbacks=[])
    trainer.test(model=model, dataloaders=build_dataloader(root_wiki+'test.csv', params_bert_rows_300_300_wikilast['embedding_dictionary']))