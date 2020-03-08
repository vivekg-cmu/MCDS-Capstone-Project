# -*- coding: utf-8 -*-
"""Capstone_bert.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1SfzsEsHuTMRHZCh631R8dFuwyALf-08C
"""

# Imports
import torch
import numpy as np

from torch import nn
import torch.nn.functional as F
from torch import optim
import pandas as pd
from tqdm import tqdm_notebook
import torch.nn.functional as F
! pip install transformers
from transformers import BertModel, BertTokenizer

# # Get more ram only for goolge colab
# temp = []
# while True:
#   temp.append(np.zeros(1000000000))

# Install The transformers package
# ! pip install transformers

# from transformers import BertModel, BertTokenizer
# # import torch

# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = BertModel.from_pretrained('bert-base-uncased')
# model = model.train()
# input_ids = torch.tensor(tokenizer.encode("Hello, my dog is as asd asf asf as", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
# # input_ids = padding_tensor(73, input_ids)
# print(input_ids.shape)
# outputs = model(input_ids)

# last_hidden_states = outputs[0]
# last_hidden_states[0][0].shape

# # Input file paths to the files
# train_file_path = '/content/d_paper_pandas.csv'
# valid_file_path = '/content/dev_paper_pandas.csv'

train_file_path = '/home/ubuntu/MCDS-Capstone-Project/social_iqa/pre_processed_datasets/train_paper_pandas.csv'
valid_file_path = '/home/ubuntu/MCDS-Capstone-Project/social_iqa/pre_processed_datasets/dev_paper_pandas.csv'

# Read Train Data
train_data = pd.read_csv(train_file_path)
valid_data = pd.read_csv(valid_file_path)



train_data.head()

max_len = 0
def tokenize_bert(tokenizer, line):
  global max_len
  ans_a = tokenizer.tokenize(line['ans_a'])
  ans_b = tokenizer.tokenize(line['ans_b'])
  ans_c = tokenizer.tokenize(line['ans_c'])

  ans_a = ans_a[:len(ans_a) - 1] + ['[PAD]' for _ in range(72 - len(ans_a))]
  ans_b = ans_b[:len(ans_b) - 1] + ['[PAD]' for _ in range(72 - len(ans_b))]
  ans_c = ans_c[:len(ans_c) - 1] + ['[PAD]' for _ in range(72 - len(ans_c))]
  input_ids_a = torch.tensor(tokenizer.encode(ans_a, add_special_tokens=True)).unsqueeze(0)  # Batch size 1
  input_ids_b = torch.tensor(tokenizer.encode(ans_b, add_special_tokens=True)).unsqueeze(0)  # Batch size 1
  input_ids_c = torch.tensor(tokenizer.encode(ans_c, add_special_tokens=True)).unsqueeze(0)  # Batch size 1
  return input_ids_a, input_ids_b, input_ids_c

def padding_tensor(padding_size, tensor):
  return F.pad(input=tensor, pad=(0, padding_size - tensor.shape[-1] + 1), value=0, mode='constant')



def get_tokenized_data(data, tokenizer):
  tokenized_dict = {
    'ans_a': [],
    'ans_b': [],
    'ans_c': [],
    }
  max_len = 0
  for i in tqdm_notebook(range(len(data))):
    ans_a, ans_b, ans_c = tokenize_bert(tokenizer, data.iloc[i])
    # max_len = max(max_len, len(ans_a[0]), len(ans_b[0]), len(ans_c))
    tokenized_dict['ans_a'].append(ans_a)
    tokenized_dict['ans_b'].append(ans_b)
    tokenized_dict['ans_c'].append(ans_c)
  # print(max_len)
  return tokenized_dict



tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
train_dict = get_tokenized_data(train_data,  tokenizer)
valid_dict = get_tokenized_data(valid_data, tokenizer)
print(max_len)

(train_dict['ans_a'][0],
train_dict['ans_b'][0],
train_dict['ans_c'][0])

cuda = True
from torch.utils import data


# Data Loader
class MyDataset(data.Dataset):
    def __init__(self, ans, y):
        self.ans = ans
        self.y = y

    def __len__(self):
        return len(self.ans['ans_a'])

    def __getitem__(self,index):
        ans_a = self.ans['ans_a'][index]
        ans_b = self.ans['ans_b'][index]
        ans_c = self.ans['ans_c'][index]
        y = self.y[index] - 1 
        return ans_a, ans_b, ans_c, y
    
    
num_workers = 8
    
# Training Loader
train_dataset = MyDataset(train_dict, train_data['one_hot_label'])

train_loader_args = dict(shuffle=True, batch_size=8, num_workers=num_workers, pin_memory=True) if cuda\
                    else dict(shuffle=True, batch_size=8)
train_loader = data.DataLoader(train_dataset, **train_loader_args)

# Validation Loader
valid_dataset = MyDataset(valid_dict, valid_data['one_hot_label'])

valid_loader_args = dict(shuffle=True, batch_size=8, num_workers=num_workers, pin_memory=True) if cuda\
                    else dict(shuffle=True, batch_size=8)
valid_loader = data.DataLoader(valid_dataset, **valid_loader_args)

for ans_a, ans_b, ans_c, y in train_loader:
  print(ans_a[:, 0, :].shape, ans_b[0], ans_c[0], y)
  break

# Simple Multilayer Perceptron
class MlpModel(torch.nn.Module):
  def __init__(self):
      super().__init__()

      self.ans_1_layer_1 = torch.nn.Linear(1024, 512)
      self.ans_1_bn1 = torch.nn.BatchNorm1d(num_features=512)
      self.ans_1_layer_2 = torch.nn.Linear(512, 256)
      self.ans_1_bn2 = torch.nn.BatchNorm1d(num_features=256)
      self.ans_1_layer_3 = torch.nn.Linear(256, 128)
      self.ans_1_layer_4 = torch.nn.Linear(128, 32)
      self.ans_1_output = torch.nn.Linear(32, 1)

      self.ans_2_layer_1 = torch.nn.Linear(1024, 512)
      self.ans_2_bn1 = torch.nn.BatchNorm1d(num_features=512)
      self.ans_2_layer_2 = torch.nn.Linear(512, 256)
      self.ans_2_bn2 = torch.nn.BatchNorm1d(num_features=256)
      self.ans_2_layer_3 = torch.nn.Linear(256, 128)
      self.ans_2_layer_4 = torch.nn.Linear(128, 32)
      self.ans_2_output = torch.nn.Linear(32, 1)

      self.ans_3_layer_1 = torch.nn.Linear(1024, 512)
      self.ans_3_bn1 = torch.nn.BatchNorm1d(num_features=512)
      self.ans_3_layer_2 = torch.nn.Linear(512, 256)
      self.ans_3_bn2 = torch.nn.BatchNorm1d(num_features=256)
      self.ans_3_layer_3 = torch.nn.Linear(256, 128)
      self.ans_3_layer_4 = torch.nn.Linear(128, 32)
      self.ans_3_output = torch.nn.Linear(32, 1)


  
  def forward(self, ans_a, ans_b, ans_c):

        
        ans_a = self.ans_1_layer_1(ans_a)
        
        ans_a = self.ans_1_bn1(ans_a)
        ans_a = F.relu(ans_a)
        ans_a = self.ans_1_layer_2(ans_a)
        ans_a = self.ans_1_bn2(ans_a)
        ans_a = F.relu(ans_a)
        ans_a = self.ans_1_layer_3(ans_a)
        ans_a = F.relu(ans_a)
        ans_a = self.ans_1_layer_4(ans_a)
        ans_a = F.relu(ans_a)
        ans_a = self.ans_1_output(ans_a)

        ans_b = self.ans_2_layer_1(ans_b)
        ans_b = self.ans_2_bn1(ans_b)
        ans_b = F.relu(ans_b)
        ans_b = self.ans_2_layer_2(ans_b)
        ans_b = self.ans_2_bn2(ans_b)
        ans_b = F.relu(ans_b)
        ans_b = self.ans_2_layer_3(ans_b)
        ans_b = F.relu(ans_b)
        ans_b = self.ans_2_layer_4(ans_b)
        ans_b = F.relu(ans_b)
        ans_b = self.ans_2_output(ans_b)

        ans_c = self.ans_3_layer_1(ans_c)
        ans_c = self.ans_3_bn1(ans_c)
        ans_c = F.relu(ans_c)
        ans_c = self.ans_3_layer_2(ans_c)
        ans_c = self.ans_3_bn2(ans_c)
        ans_c = F.relu(ans_c)
        ans_c = self.ans_3_layer_3(ans_c)
        ans_c = F.relu(ans_c)
        ans_c = self.ans_3_layer_4(ans_c)
        ans_c = F.relu(ans_c)
        ans_c = self.ans_3_output(ans_c)

        output = torch.cat((ans_a, ans_b, ans_c), dim=1)
        

        return output

class Model(torch.nn.Module):
  def __init__(self, bert_model, mlp_model):
      super().__init__()
      self.bert_model = bert_model
      self.mlp_model = mlp_model

  def forward(self, ans_a, ans_b, ans_c):
        # ans1 = torch.mean(self.bert_model(ans_a)[0],1)
        # ans2 = torch.mean(self.bert_model(ans_b)[0], 1)
        # ans3 = torch.mean(self.bert_model(ans_c)[0], 1)


        ans1 = self.bert_model(ans_a)[0][:, 0, :]
        ans2 = self.bert_model(ans_b)[0][:, 0, :]
        ans3 = self.bert_model(ans_c)[0][:, 0, :]

        output = self.mlp_model(ans1, ans2, ans3)
        
        return output

temp = BertModel.from_pretrained('bert-large-uncased')

# Get Model
torch.cuda.empty_cache()
model = Model(temp, MlpModel())
model = model.cuda()

# Cross Entropy Function
loss_function = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=0,verbose=True, min_lr=1e-6)

# Training Code
from tqdm import tqdm_notebook

def compute_loss(x_output, labels):
    np_y = np.array(labels.cpu().data)
    np_x = np.argmax(x_output.cpu().data.numpy(), axis=1)
    result = np.sum(np.array([1 for x, y in zip(np_x, np_y) if x == y]))
    return (result/len(np_y))

# for p in model.bert_model.parameters():
#   if p.requires_grad:
#       old = p.cpu().data.numpy()
#       break
# old = list(model.bert_model.parameters())[-1]

for epoch in range(1, 100): ## run the model for 10 epochs
    train_loss, valid_loss = [], []
    model = model.train()
    # new = list(model.bert_model.parameters())[-1]
    # print(torch.eq(old , new).all())
    # old = list(model.bert_model.parameters())[-1]


    accuracy = []
    
    pbar = tqdm_notebook(total=len(train_dict['ans_a']))

    for batch_id, (ans_a, ans_b, ans_c, target) in enumerate(train_loader):
        
        ans_a = ans_a[:,0, :].cuda()
        ans_b = ans_b[:,0, :].cuda()
        ans_c = ans_c[:,0, :].cuda()
        target = target.cuda()
        pbar.update(8)
        optimizer.zero_grad()
        
        ## 1. forward propagation
        output = model(ans_a, ans_b, ans_c)
        res = compute_loss(output, target)
        accuracy.append(res)
        ## 2. loss calculation
        loss = loss_function(output, target)
        
        
        ## 3. backward propagation
        loss.backward()
        
        ## 4. weight optimization
        optimizer.step()
        
        train_loss.append(loss.item())
    print("Training Accuracy:", sum(accuracy)/len(accuracy))
    
    # model.eval()
    # valid_acc = []
    # for batch_id, (ans_a, ans_b, ans_c, target) in enumerate(valid_loader):
    #     print(ans_a.shape)
    #     ans_a = ans_a[:,0, :].cuda()
    #     print(ans_a.shape)
    #     ans_b = ans_b[:,0, :].cuda()
    #     ans_c = ans_c[:,0, :].cuda()
    #     target = target.cuda()

        
    #     output = model(ans_a, ans_b, ans_c)
    #     loss = loss_function(output, target)
    #     res_valid = compute_loss(output, target)
    #     valid_acc.append(res_valid)
    #     valid_loss.append(loss.item())
    # print("Validation Accuracy:", sum(valid_acc)/len(valid_acc))
    print ("Epoch:", epoch, "Training Loss: ", np.mean(train_loss), "Valid Loss: ", np.mean(train_loss))
    scheduler.step(np.mean(train_loss))
    # print('-----------------------------------------------------------------------')

tokenize_bert("hello how are you doing")

ans_a.shape
train_dict

model.bert_model(

i = 0
for elem in train_loader:
  print(i)
  i += 1

len(train_loader)

train_data['one_hot_label']

x = []
y = []
for p in model.bert_model.parameters():
    if p.requires_grad:
         x.append(p)
    y.append(p)   
len(x), len(y)

from torchviz import make_dot

import torch
from torchviz import make_dot

x = []
for p in model.bert_model.parameters():
    if p.requires_grad:
         x.append(p)

model.bert_model.encoder

model

tokenizer.encode("[CLS] hello how [PAD] [PAD] [SEP] hi bro [PAD] [PAD] [SEP] [PAD]")

