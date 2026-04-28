import sys
from lightning.pytorch import seed_everything
seed_everything(42, workers=True)
sys.path.append(".")
sys.path.append("../../")
import pickle
from Baselines._2_train_scaling_models import *
from Baselines._1_generate_table_embeddings import *

def only_MAE(model_name: str, model_checkpoint: str, embedding_path: str|dict, test_dataset_path: str) -> dict[str,float]:
    trainer_config =   {
        'accelerator': "gpu",
        'fast_dev_run' : False, 
        'deterministic' : True, 
        'devices':[1],
        'min_epochs':1,
        'max_epochs':100,
        'log_every_n_steps':50,
    }

    model = Embedding_scaler.load_from_checkpoint(model_checkpoint)
    trainer =  L.Trainer(**trainer_config, callbacks=[])
    if isinstance(embedding_path, str):
        with open(embedding_path, 'rb') as f:
            embedding_path = pickle.load(f)
    if isinstance(test_dataset_path, str):
        test_dataset_path = pd.read_csv(test_dataset_path)
    start = time.time()
    if model_name == 'embdi_on_wiki':
        out = trainer.test(model=model, dataloaders=build_dataloader(test_dataset_path, embedding_path, batch_size=1, is_embdi=True))
    elif model_name == 'turl_t_g_on_git':
        out = trainer.test(model=model, dataloaders=build_dataloader(test_dataset_path, embedding_path, batch_size=1, remove_missing_pairs=True))
    else:
        out = trainer.test(model=model, dataloaders=build_dataloader(test_dataset_path, embedding_path, batch_size=1))
    end = time.time()
    print(f'Completed in {end-start} seconds')
    return {model_name: {'mae':out[0]['test_loss'], 'inference_time':(end-start)}}

root = ''
root_git = root+'/GitTables/'
root_wiki = root+'/WikiTables/'

models_git= root_git+'/models/'
embeddings_git = root_git+'dictionaries/embedding_dictionaries/'
test_git = root_git+'test.csv'

models_wiki = root_wiki+'/models/'
embeddings_wiki = root_wiki+'dictionaries/embedding_dictionaries/'
test_wiki = root_wiki+'test.csv'

results_path = root+'/evaluation/'

if __name__ == '__main__':
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    #EMBDI
    # inference wiki
    embdi_on_wiki = {
        'model_name':'embdi_on_wiki',
        'model_checkpoint':models_wiki+'/embdi_wikilast.pth',
        'embedding_path':embeddings_wiki+'/emb_dict_embdi.pkl',
        'test_dataset_path':test_wiki
    }

    #TURL-T-G
    # inference git
    turl_t_g_on_git = {
        'model_name':'turl_t_g_on_git',
        'model_checkpoint':models_git+'/turl_tables_300_300_gittables.pth',
        'embedding_path':embeddings_git+'/emb_dict_turl_tables_128_128.pkl',
        'test_dataset_path':test_git
    }

    #TURL-T-W
    # inference wiki
    turl_t_w_on_wiki = {
        'model_name':'turl_t_w_on_wiki',
        'model_checkpoint':models_wiki+'/turl_tables_300_300_wikilast.pth',
        'embedding_path':embeddings_wiki+'/emb_dict_turl_tables_300_300.pkl',
        'test_dataset_path':test_wiki
    }

    #BERT-R-G
    # inference git
    bert_r_g_on_git = {
        'model_name':'bert_r_g_on_git',
        'model_checkpoint':models_git+'/bert_rows_300_300_gittables.pth',
        'embedding_path':embeddings_git+'/emb_dict_bert_lines_300_300.pkl',
        'test_dataset_path':test_git
    }

    #BERT-R-W
    # inference wiki
    bert_r_w_on_wiki = {
        'model_name':'bert_r_w_on_wiki',
        'model_checkpoint':models_wiki+'/bert_rows_300_300_wikilast.pth',
        'embedding_path':embeddings_wiki+'/emb_dict_bert_lines_300_300.pkl',
        'test_dataset_path':test_wiki
    }

    #BERT-T-G
    # inference git
    bert_t_g_on_git = {
        'model_name':'bert_t_g_on_git',
        'model_checkpoint':models_git+'/bert_tables_300_300_gittables.pth',
        'embedding_path':embeddings_git+'/emb_dict_bert_tables_300_300.pkl',
        'test_dataset_path':test_git
    }
    # inference wiki
    bert_t_g_on_wiki = {
        'model_name':'bert_t_g_on_wiki',
        'model_checkpoint':models_git+'/bert_tables_300_300_gittables.pth',
        'embedding_path':embeddings_wiki+'/emb_dict_bert_tables_300_300.pkl',
        'test_dataset_path':test_wiki
    }

    #BERT-T-W
    # inference wiki
    bert_t_w_on_wiki = {
        'model_name':'bert_t_w_on_wiki',
        'model_checkpoint':models_wiki+'/bert_tables_300_300_wikilast.pth',
        'embedding_path':embeddings_wiki+'/emb_dict_bert_tables_300_300.pkl',
        'test_dataset_path':test_wiki
    }
    # inference git
    bert_t_w_on_git = {
        'model_name':'bert_t_w_on_git',
        'model_checkpoint':models_wiki+'/bert_tables_300_300_wikilast.pth',
        'embedding_path':embeddings_git+'/emb_dict_bert_tables_300_300.pkl',
        'test_dataset_path':test_git
    }

    #BERT-T-N-W
    # inference wiki
    bert_t_n_w_on_wiki = {
        'model_name':'bert_t_n_w_on_wiki',
        'model_checkpoint':models_wiki+'/bert_tables_anon_300_300.pth',
        'embedding_path':embeddings_wiki+'/emb_dict_bert_tables_anon_300_300.pkl',
        'test_dataset_path':test_wiki
    }

    #ROBERTA-R-G
    # inference git
    roberta_r_g_on_git = {
        'model_name':'roberta_r_g_on_git',
        'model_checkpoint':models_git+'/roberta_rows_300_300_gittables.pth', 
        'embedding_path':embeddings_git+'/emb_dict_roberta_lines_300_300.pkl',
        'test_dataset_path':test_git
    } 
    #ROBERTA-R-W
    # inference wiki
    roberta_r_w_on_wiki = {
        'model_name':'roberta_r_w_on_wiki',
        'model_checkpoint':models_wiki+'/roberta_rows_300_300_wikilast.pth',
        'embedding_path':embeddings_wiki+'/emb_dict_roberta_lines_300_300.pkl',
        'test_dataset_path':test_wiki
    }

    #ROBERTA-T-G
    # inference git
    roberta_t_g_on_git = {
        'model_name':'roberta_t_g_on_git',
        'model_checkpoint':models_git+'/roberta_tables_300_300_gittables.pth',
        'embedding_path':embeddings_git+'/emb_dict_roberta_tables_300_300.pkl',
        'test_dataset_path':test_git
    }
    # inference wiki
    roberta_t_g_on_wiki = {
        'model_name':'roberta_t_g_on_wiki',
        'model_checkpoint':models_git+'/roberta_tables_300_300_gittables.pth',
        'embedding_path':embeddings_wiki+'/emb_dict_roberta_tables_300_300.pkl',
        'test_dataset_path':test_wiki
    }

    #ROBERTA-T-W
    # inference wiki
    roberta_t_w_on_wiki = {
        'model_name':'roberta_t_w_on_wiki',
        'model_checkpoint':models_wiki+'/roberta_tables_300_300_wikilast.pth',
        'embedding_path':embeddings_wiki+'/emb_dict_roberta_tables_300_300.pkl',
        'test_dataset_path':test_wiki
    }
    # inference git
    roberta_t_w_on_git = {
        'model_name':'roberta_t_w_on_git',
        'model_checkpoint':models_wiki+'/roberta_tables_300_300_wikilast.pth',
        'embedding_path':embeddings_git+'/emb_dict_roberta_tables_300_300.pkl',
        'test_dataset_path':test_git
    }

    #ROBERTA-T-N-W
    # inference wiki
    roberta_t_n_w_on_wiki = {
        'model_name':'roberta_t_n_w_on_wiki',
        'model_checkpoint':models_wiki+'/roberta_tables_anon_300_300.pth',
        'embedding_path':embeddings_wiki+'/emb_dict_roberta_tables_anon_300_300.pkl',
        'test_dataset_path':test_wiki
    }

    tests = [
        embdi_on_wiki,
        turl_t_g_on_git,
        turl_t_w_on_wiki,
        bert_r_g_on_git,
        bert_r_w_on_wiki,
        bert_t_g_on_git,
        bert_t_g_on_wiki,
        bert_t_w_on_wiki,
        bert_t_w_on_git,
        bert_t_n_w_on_wiki,
        roberta_r_g_on_git,
        roberta_r_w_on_wiki,
        roberta_t_g_on_git,
        roberta_t_g_on_wiki,
        roberta_t_w_on_wiki,
        roberta_t_w_on_git,
        roberta_t_n_w_on_wiki
    ]
    results = {}
    for t in tests:
        print(f'Starting {t["model_name"]}')
        results.update(only_MAE(**t))
    print(results)

    with open(results_path,'wb') as f:
        pickle.dump(results,f)