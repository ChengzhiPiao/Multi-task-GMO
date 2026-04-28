import torch
from transformers import BertModel, BertTokenizer
from abc import ABC, abstractmethod
import gensim.downloader as api
import hashlib
import pandas as pd

class Embedding_buffer(ABC):
    @abstractmethod
    def __call__(self, sentence: str) -> NotImplemented: # type: ignore
        """Adds a sentence to the buffer of sentences to embed

        Args:
            sentence (str): the sentence to embed

        Returns:
            NotImplemented: up to the specific implementation
        """
        return NotImplemented
    
    @abstractmethod
    def pop_embeddings(self) -> NotImplemented: # type: ignore
        """The buffer is emptied and the embeddings inside it returned

        Returns:
            NotImplemented: up to the specific implementation
        """
        return NotImplemented
    
    @abstractmethod
    def add_nan_embedding(self) -> NotImplemented: # type: ignore
        """Special method to manage nan sentences

        Returns:
            NotImplemented: up to the specific implementation
        """
        return NotImplemented

class Hash_embedding_buffer(Embedding_buffer):
    def __init__(self) -> None:   #'fasttext-wiki-news-subwords-300'
        """The class init method

        Args:
            model (str, optional): The version of fasttext/word2vec to use {'fasttext-wiki-news-subwords-300', 'word2vec-google-news-300'}. Defaults to 'word2vec-google-news-300'.
        """
        self.vector_size = 32
        self.n_embeddings = 0
        self.embeddings = None

    def add_nan_embedding(self) -> None:
        """Method to manage the nan values
        """
        vector = torch.zeros(self.vector_size, dtype=torch.float)
        try:
            self.embeddings = torch.cat((self.embeddings, vector.unsqueeze(0)), dim=0)
        except TypeError:
            self.embeddings = vector.unsqueeze(0)
        self.n_embeddings += 1

    def add_random_embedding(self) -> None:
        vector = torch.rand(self.vector_size, dtype=torch.float)
        try:
            self.embeddings = torch.cat((self.embeddings, vector.unsqueeze(0)), dim=0)
        except TypeError:
            self.embeddings = vector.unsqueeze(0)
        self.n_embeddings += 1

    def __get_embedding(self, word: str) -> torch.Tensor:
        """Provide the embedding of a word

        Args:
            word (str): the word to embed

        Returns:
            torch.Tensor: the embedding of the word
        """
        hashobj = hashlib.sha256(word.encode())
        hex_string = hashobj.hexdigest()
        
        chunk_size = 2
        chunks = [hex_string[i:i + chunk_size] for i in range(0, len(hex_string), chunk_size)]

        embedding = [int(chunk, 16) for chunk in chunks]

        return embedding

    def __call__(self, sentence: str) -> None:
        """Add the sentence to embed to the buffer

        Args:
            sentence (str): sentence to embed
        """
        emb = torch.tensor(self.__get_embedding(sentence), dtype = torch.float)
        self.n_embeddings += 1
        try:
            self.embeddings = torch.cat((self.embeddings, emb.unsqueeze(0)), dim=0)
        except TypeError:
            self.embeddings = emb.unsqueeze(0) 

    def pop_embeddings(self) -> torch.Tensor:
        """Return all the generated embeddings and reset the buffer

        Returns:
            torch.Tensor: tensor with one row for every embedding 
        """
        out = self.embeddings
        self.embeddings = None
        self.n_embeddings = 0
        return out

class BertEmbeddingBuffer(Embedding_buffer):
    def __init__(self,
                 max_lines: int = 128,
                 max_columns = 128,
                 gpu_num: str='0') -> None:
        self.gpu_num = gpu_num
        self.device = ('cuda:'+self.gpu_num) if torch.cuda.is_available() else 'cpu'
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=False).to(self.device).eval()
        self.max_lines = max_lines
        self.max_columns = max_columns
        self.vector_size = 768
    
    def prepare_sentence(self, df: pd.DataFrame) -> str:
        out = ''
        for r in range(min(df.shape[0], self.max_lines)):
            row = []
            for c in range(min(df.shape[1], self.max_columns)):
                val = str(df.iloc[r].iloc[c])
                val = val.replace(',', '')
                val = val.replace('#', '')
                if val == '':
                    val = 'NULL'
                row.append(val)
            out = out + ','.join(row) + '#'
        return out
    def get_max_rows_columns(self, r, c) -> tuple:
        max_cols = min(c, self.max_embedding_index)
        max_rows = min(r, self.max_embedding_index//max_cols)
        return max_rows, max_cols
    
    def load_table(self, df: pd.DataFrame) -> None:
        # Keys: ,=1010, #=1001
        self.n_embeddings = 0
        self.embeddings = None
        sentence = self.prepare_sentence(df)
        enc = self.tokenizer(sentence, padding=True, truncation=True, max_length=512, add_special_tokens=False)
        enc = {k:torch.LongTensor(v).unsqueeze(0).to(self.device) for k, v in enc.items()}
        mappings = enc['input_ids'].squeeze(0)
        with torch.no_grad():
            out = self.model(**enc)
        hidden_states = out['last_hidden_state']
        token_embeddings = hidden_states.squeeze(0).cpu()
        self.embeddings_unstacked = []
        buffer = torch.zeros(self.vector_size, dtype=torch.float)
        for i in range(token_embeddings.shape[0]):
            if mappings[i] == 1001 or mappings[i] == 1010:
                self.embeddings_unstacked.append(buffer)
                buffer = torch.zeros(self.vector_size, dtype=torch.float)
                continue
            buffer += token_embeddings[i]
        self.next_embedding_index = 0
        if len(self.embeddings_unstacked)==0:
            print('Table too large, returning empty tensor')
            self.embeddings_unstacked.append(torch.zeros(self.vector_size, dtype=torch.float))
        self.max_embedding_index = len(self.embeddings_unstacked)

    def add_nan_embedding(self) -> None:
        """Method to manage the nan values
        """
        self.__call__()

    def add_random_embedding(self) -> None:
        self.__call__()

    def __get_embedding(self) -> torch.Tensor | str:
        if self.next_embedding_index >= self.max_embedding_index:
            if self.embeddings == None:
                print('Table too large, returning empty tensor')
                self.embeddings = torch.zeros(self.vector_size, dtype=torch.float).unsqueeze(0)
            raise IndexError('No more embeddings to return')
        emb = self.embeddings_unstacked[self.next_embedding_index]
        self.next_embedding_index += 1
        return emb

    def __call__(self) -> None:
        emb = self.__get_embedding()
        self.n_embeddings += 1
        try:
            self.embeddings = torch.cat((self.embeddings, emb.unsqueeze(0)), dim=0)
        except TypeError:
            self.embeddings = emb.unsqueeze(0) 

    def pop_embeddings(self) -> torch.Tensor:
        """Return all the generated embeddings and reset the buffer

        Returns:
            torch.Tensor: tensor with one row for every embedding 
        """
        out = self.embeddings
        self.embeddings = None
        self.n_embeddings = 0
        return out


if __name__ == '__main__':
    # Example of use
    df = pd.DataFrame({'a': ['geronimo, stilton, e#paperino', '', 'magic the gathering 2'], 'b': [4, 5, 6]})
    emb = BertEmbeddingBuffer()
    emb.load_table(df)
    emb.add_nan_embedding()
    emb.add_random_embedding()
    emb('Hello world')
    