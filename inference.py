import os, csv, argparse
import numpy as np
import torch
from transformers import AdamW, BertModel
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from copy import deepcopy
from tqdm import tqdm, trange
from data import CorpusData
from head import My_linear

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=0)
    parser.add_argument('--mode', type=str, default='test')
    return parser.parse_args()

args = _parse_args()

bert_model = BertModel.from_pretrained("bert-base-uncased")
bert_model.load_state_dict(torch.load("model/e{}_bert.pkl".format(args.epoch)))
bert_model.to(device)
linear_model = My_linear()
linear_model.load_state_dict(torch.load("model/e{}_linear.pkl".format(args.epoch)))
linear_model.to(device)

history = {'training':[],'validation':[],'debug':[]}

def train_epoch(curr_db, description, batch_size=1):
    if description.lower() is 'training':
        shuffle=True
    else:
        shuffle=False
    
    mode = description.lower()
    assert mode in ['training', 'validation', 'debug'], 'Unknown mode {}'.format(description)
    dataloader = DataLoader(dataset=curr_db, batch_size=batch_size, shuffle=shuffle, num_workers=0)
    if mode == 'validation':
        bert_model.eval()
        linear_model.eval()
    else:
        bert_model.train()
        linear_model.train()

    with open('output/submission_{}.csv'.format(args.epoch), 'w', newline='') as csvfile:
        fieldnames = ['Index', 'Gold']

        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for bi, (padded, segment, attention_mask, seq_id) \
            in tqdm(enumerate(dataloader), desc=description, total=len(dataloader)):

            token_tensor = padded.to(device)
            segment = segment.to(device)
            mask = attention_mask.to(device)
            
            if mode is 'validation':
                with torch.no_grad():
                    last_hidden, pooler_output = bert_model(token_tensor, token_type_ids=segment, attention_mask=mask)
            else:
                last_hidden, pooler_output = bert_model(token_tensor, token_type_ids=segment, attention_mask=mask)
        
            cls_score = linear_model(pooler_output)
            cls_score = torch.sigmoid(cls_score)
            cls_score = 1 if cls_score.view(1).item() > 0.94 else 0

            #writer.writerow({'Index': seq_id[0][0], 'Gold': cls_score.view(1).item()})
            writer.writerow({'Index': seq_id[0][0], 'Gold': cls_score})


# train_db=CorpusData(partition='train')
test_db=CorpusData(partition='test')
valid_db=CorpusData(partition='valid')

train_epoch(test_db, 'validation', batch_size=1)
