import os, csv
import numpy as np
import torch
from transformers import BertTokenizer
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class CorpusData(Dataset):
    def __init__(self, MAX=512, partition='train'):
        bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        if partition is 'train':
            f_name = 'resources/train.csv'
        elif partition is 'test':
            f_name = 'resources/test.csv'

        with open(f_name, 'r') as f:
            self._data = list(csv.DictReader(f, delimiter=';'))
            for row in self._data:
                row['Text'] = '[CLS]' + row['Text'] + '[SEP]'
                # Split the sentence into tokens.
                row['tokenized_text'] = bert_tokenizer.tokenize(row['Text'])        
                # Map the token strings to their vocabulary indeces.
                row['indexed_tokens'] = bert_tokenizer.convert_tokens_to_ids(row['tokenized_text'])
        
        self.max_len = 0
        for item in self._data:
            if len(item['indexed_tokens']) > self.max_len:
                self.max_len = len(item['indexed_tokens'])

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index):
        padded = np.array([self._data[index]['indexed_tokens'] + [0]*(self.max_len-len(self._data[index]['indexed_tokens'])))
        segment = np.zeros(self.max_len)
        attention_mask = np.where(padded != 0, 1, 0)
        seq_id = np.array([self._data[index]['Index']])
        gold = np.array([self._data[index]['Gold']])

        return padded, segment, attention_mask, seq_id, gold
        