from torch.utils.data import Dataset
import pandas as pd

class Embedding_dataset(Dataset):
    def __init__(self, triple_dataset: pd.DataFrame, embedding_dictionary: dict) -> None:
        super().__init__()
        self.embedding_dictionary = embedding_dictionary
        self.triple_dataset = triple_dataset
    
    def __len__(self) -> int:
        return self.triple_dataset.shape[0]
    
    def __getitem__(self, index) -> tuple:
        er = self.embedding_dictionary[self.triple_dataset.iloc[index].loc['r_id']]
        es = self.embedding_dictionary[self.triple_dataset.iloc[index].loc['s_id']]
        label = self.triple_dataset.iloc[index].loc['a%']
        return er, es, label

class Embedding_dataset_embdi(Dataset):
    def __init__(self, triple_dataset: pd.DataFrame, embedding_dictionary: dict) -> None:
        super().__init__()
        self.embedding_dictionary = embedding_dictionary
        self.triple_dataset = triple_dataset
    
    def __len__(self) -> int:
        return self.triple_dataset.shape[0]
    
    def __getitem__(self, index) -> tuple:
        r_id = self.triple_dataset.iloc[index].loc['r_id']
        s_id = self.triple_dataset.iloc[index].loc['s_id']
        k = f'{r_id}|{s_id}'
        er = self.embedding_dictionary[k][0]
        es = self.embedding_dictionary[k][1]
        label = self.triple_dataset.iloc[index].loc['a%']
        return er, es, label