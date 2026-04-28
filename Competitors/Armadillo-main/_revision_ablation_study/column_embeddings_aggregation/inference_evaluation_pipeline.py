import sys
sys.path.append(".")
sys.path.append("../../")

from _revision_ablation_study.column_embeddings_aggregation._generate_graph_dict import *
from _revision_ablation_study.column_embeddings_aggregation.armadillo import Armadillo
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Dataset, Batch
import torch.nn.functional as F
from tqdm import tqdm
import torch
import pickle
import time
import os

class GraphDataset(Dataset):
    
    def __init__(self, graphs: dict = None) -> None:
        """Init function

        Args:
            triples (pd.DataFrame): Dataframe that contains triples ()'r_id','s_id','table_overlap')
            graphs (dict): a dictionary containing a graph for every key that appears in the triples dataset
            tables (Optional[dict], optional): not implemented. Defaults to None.
        """
        super(GraphDataset, self).__init__()
        self.graphs = [(k,graphs[k]) for k in graphs.keys()]

    def len(self) -> int:
        return len(self.graphs)
    
    def get(self, idx:int) -> tuple:
        k = self.graphs[idx][0]
        g = self.graphs[idx][1]
        return Data(g.X, g.edges), k, g.next_column_index, g.next_row_index, g.X.shape[0]

def embed_all(graph_dict: dict|str, model_path: str, batch_size: int=64, to_cpu: bool=False, num_workers: int=0) -> dict:
    if isinstance(graph_dict, str):
        with open(graph_dict,'rb') as f:
            graph_dict = pickle.load(f)
    
    dataset = GraphDataset(graphs=graph_dict)
    loader = DataLoader(dataset, 
                             batch_size=batch_size,  
                             num_workers=num_workers, 
                             shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Armadillo(model_file=model_path)
    model.eval()
    embeddings_dict = {}
    with torch.no_grad():
        for batch in tqdm(loader):
            embeddings = model(batch[0].to(device), batch[2], batch[4])
            for i, k in enumerate(batch[1]):
                if to_cpu:
                    embeddings_dict[k] = embeddings[i].detach().cpu()
                else:  
                    embeddings_dict[k] = embeddings[i].detach()
    return embeddings_dict

def compute_mae(embeddings_dict: str|dict, test_dataset: str|pd.DataFrame) -> float:
    if isinstance(test_dataset, str):
        test_dataset = pd.read_csv(test_dataset)
    if isinstance(embeddings_dict, str):
        with open(embeddings_dict,'rb') as f:
            embeddings_dict = pickle.load(f)
    errors = []
    for r in tqdm(range(test_dataset.shape[0])):
        r_id = test_dataset.iloc[r]['r_id']
        s_id = test_dataset.iloc[r]['s_id']
        overlap_true = test_dataset.iloc[r]['a%']
        cos_sim = float(F.cosine_similarity(embeddings_dict[r_id],embeddings_dict[s_id],dim=0))
        overlap_pred = max(cos_sim,0)
        errors.append(abs(overlap_true-overlap_pred))
    mae = sum(errors)/len(errors)
    print(f"MAE: {mae}")
    return mae

if __name__ == '__main__':
    root_ablation = '/home/francesco.pugnaloni/armadillo_all/ablation_study/column_aggreg/'
    root_dataset = '/home/francesco.pugnaloni/armadillo_all/datasets/WikiTables/'
    table_dict = root_dataset+'/dictionaries/table_dict.pkl'
    test_dataset = root_dataset+'/test.csv'
    model_path = root_ablation+'/models/armadillo.pth'

    df = pd.read_csv(test_dataset)

    table_set = set(df['r_id']).union(set(df['s_id']))

    with open(table_dict,'rb') as f:
        table_dict = pickle.load(f)
    table_dict = {k:table_dict[k] for k in table_set}
    print('Building graph_dict')
    start_dict = time.time()
    graph_dict = generate_graph_dictionary(table_dict_path=table_dict, embedding_generation_method='sha256', save_graph_dict=False)
    end_dict = time.time()
    graph_dict_time = end_dict-start_dict
    with open(root_ablation+'/log/graph_dict_gen_time_evaluation.txt','w') as f:
        f.write(str(graph_dict_time)+' sec')
    print('Graph dictionary generated')

    print('Embedding generation satarting')
    start_emb = time.time()
    embeddding_dict = embed_all(graph_dict, model_path)
    end_emb = time.time()
    emb_time = end_emb-start_emb
    with open(root_ablation+'/log/emb_time_evaluation.txt','w') as f:
        f.write(str(emb_time)+' sec')

    print('Overlap computation starting')
    start_overlap = time.time()
    mae = compute_mae(embeddding_dict, test_dataset)
    end_overlap = time.time()
    overlap_time = end_overlap-start_overlap
    with open(root_ablation+'/log/overlap_time_evaluation.txt','w') as f:
        f.write(str(overlap_time)+' sec ---- MAE: '+str(mae))