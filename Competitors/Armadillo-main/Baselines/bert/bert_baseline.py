import torch
from transformers import BertModel, BertTokenizer, RobertaModel, RobertaTokenizer
import pandas as pd
import csv
import os
from tqdm import tqdm

def prepare_sentence_list(table: pd.DataFrame, max_lines: int, max_columns: int, granularity: str) -> list[str]:
    out = []
    for r in range(min(table.shape[0], max_lines)):
        row = []
        for c in range(min(table.shape[1], max_columns)):
            row.append(str(table.iloc[r].iloc[c]))
        if granularity == 'row':
            out.append(", ".join(row))
        elif granularity == 'cell':
            out += row
        elif granularity == 'table':
            if len(out) == 0:
                out.append('')
                out[0] = out[0] + ", ".join(row)
            else:
                out[0] = out[0] + '\n' + ", ".join(row)
    return out

class Bert_table_embedder:

    def __init__(self,
                 max_lines: int = 128,
                 max_columns = 128,
                 output_format: str = 'mean',
                 granularity: str = 'row',
                 output_hidden_states: bool=False, 
                 model: str='bert',
                 gpu_num: str='0') -> None:

        self.device = ('cuda:'+gpu_num) if torch.cuda.is_available() else 'cpu'
        if model == 'bert':
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=output_hidden_states).to(self.device).eval()
        elif model == 'roberta':
            self.tokenizer = RobertaTokenizer.from_pretrained('FacebookAI/roberta-base') 
            self.model = RobertaModel.from_pretrained('FacebookAI/roberta-base', output_hidden_states=output_hidden_states).to(self.device).eval()
        else:
            raise Exception('Model not supported')
        self.granularity = granularity
        self.max_lines = max_lines
        self.max_columns = max_columns
        self.output_format = output_format

    def encode(self, l: list[str]) -> list:
        try:
            return self.tokenizer(l, padding=True, truncation=True, max_length=self.max_columns, add_special_tokens=True)
        except:
            l = ['']
            return self.tokenizer(l, padding=True, truncation=True, max_length=self.max_columns, add_special_tokens=True)

    def __call__(self, table: pd.DataFrame) -> torch.Tensor:
        sentence_list = prepare_sentence_list(table, max_lines=self.max_lines, max_columns=self.max_columns, granularity=self.granularity)
        embeddings = []
        for s in sentence_list:
            embeddings.append(self.embed_sentence(s))
        return (sum(embeddings) / len(embeddings)).cpu()

    def embed_sentence(self, sentence: str, strategy: str='CLS') -> torch.Tensor:
        enc = self.encode(sentence)
        enc = {k:torch.LongTensor(v).unsqueeze(0).to(self.device) for k, v in enc.items()}
        with torch.no_grad():
            out = self.model(**enc)
        hidden_states = out['last_hidden_state']

        if strategy == 'CLS':
            sentence_embedding = hidden_states[:,0]

        elif strategy == 'average':
            sentence_embedding = torch.mean(hidden_states, dim=1)
        
        if len(sentence) == 1:
            return sentence_embedding.squeeze(0)
        else:
            return sentence_embedding

if __name__ == '__main__':
    table_path = '/home/francesco.pugnaloni/armadillo_all/datasets/gittables/csv/train_csv/abstraction_csv_licensed.zip_00-01_37.csv'
    # table_path = '/data/survey/csv/zeolite4A.csv'
    
    table = pd.read_csv(table_path)
    
    sent = prepare_sentence_list(table, max_lines=300, max_columns=300, granularity='table')

    tt = Bert_table_embedder(granularity='row')
    val = tt(table)
    print('end')
    