import os, csv
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

bert_model = BertModel.from_pretrained("bert-base-uncased")
bert_model.to(device)
linear_model = My_linear()
linear_model.to(device)

cls_criteria = nn.BCELoss().to(device)
params = list(bert_model.parameters()) + list(linear_model.parameters())

total_optimizer = AdamW(params, lr=5e-05, weight_decay=0.012)

max_epoch = 10
START_epoch = 0

history = {'training':[],'validation':[],'debug':[]}

def train_epoch(curr_db, description, batch_size=32):
    if description.lower() is 'training':
        shuffle=True
    else:
        shuffle=False
    
    mode = description.lower()
    assert mode in ['training', 'validation', 'debug'], 'Unknown mode {}'.format(description)
    dataloader = DataLoader(dataset=curr_db, batch_size=batch_size, shuffle=shuffle, collate_fn=curr_db.collate_fn, num_workers=0)
    if mode == 'validation':
        bert_model.eval()
        linear_model.eval()
    else:
        bert_model.train()
        linear_model.train()
    
    epoch_total_loss=0
    epoch_cls_loss=0

    num_of_batch = len(dataloader)
    counter = 0.333

    for bi, (padded, segment, attention_mask, seq_id, gold) \
        in tqdm(enumerate(dataloader), desc=description, total=len(dataloader)):

        token_tensor = torch.from_numpy(padded).to(device)
        segment = torch.from_numpy(segment).to(device)
        mask = torch.from_numpy(attention_mask).to(device)
        Y_label = torch.from_numpy(gold).to(device)
        
        total_optimizer.zero_grad()
        if mode == 'validation':
            with torch.no_grad():
                last_hidden, pooler_output = bert_model(token_tensor, token_type_ids=segment, attention_mask=mask)
        else:
            last_hidden, pooler_output = bert_model(token_tensor, token_type_ids=segment, attention_mask=mask)
    
        cls_score = linear_model(pooler_output, "cls")
        cls_loss = cls_criteria(cls_score, Y_label)
        
        total_loss = cls_loss

        if mode != 'validation':
            total_loss.backward()
            total_optimizer.step()
        else:
            pass
        
        cls_score = cls_score.cpu().detach()


        batch_size = len(qids)
        for i in range(batch_size):
            predictions[mode]["{}".format(qids[i])] = answers[i].replace(" ","")

        epoch_total_loss += total_loss/len(curr_db)

        if mode in ['training'] and bi/num_of_batch > counter: #breakpoint saving
            save_train(epoch, 'f'+str(counter))
            print('Saved at breakpoint: epoch {}, frac {}, where batch_loss={}'.format(epoch, counter, total_loss))
            counter += 0.334
    
    epoch_loss_log = "Epoch:{} | total_loss: {}".format(epoch, epoch_total_loss)
    history[mode].append(epoch_loss_log)
            

def save_train(epoch, flag=False, save_model=True):
    if not os.path.exists('model'):
        os.mkdir('model')
    if not os.path.exists('predictions'):
        os.mkdir('predictions')
    if not os.path.exists('history'):
        os.mkdir('history')
    if flag and save_model:
        epoch = str(epoch)+'_{}'.format(flag)
        torch.save(linear_model.state_dict(), 'model/e{}_linear.pkl'.format(epoch))
        torch.save(bert_model.state_dict(), 'model/e{}_bert.pkl'.format(epoch))
        return

    if save_model:
        torch.save(linear_model.state_dict(), 'model/e{}_linear.pkl'.format(epoch))
        torch.save(bert_model.state_dict(), 'model/e{}_bert.pkl'.format(epoch))

    for mode in history:
        with open(r'history/history_' + mode + '.txt', 'w', encoding='utf-8') as f:
            for his in history[mode]:
                f.write(str(his)+'\n')
    for mode in predictions:

        with open(r'predictions/predict_' + mode + '_e{}_'.format(epoch) + '.json', 'w', encoding='utf-8') as f:
            f.write(json.dumps(predictions[mode]))

train_db=CorpusData(partition='train')
# valid_db=CorpusData(partition='test')

for epoch in range(max_epoch):
    epoch += START_epoch
    print('Epoch: {}'.format(epoch))
    predictions = {'training':{}, 'validation':{},'debug':{}} 
    #total_optimizer.zero_grad()
    try:
        train_epoch(train_db, 'Training', batch_size=8)
        # train_epoch(valid_db, 'validation', batch_size=4)
    except:
        # save_train(epoch, 'WARN')
        raise
    save_train(epoch)