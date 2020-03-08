# -*- coding: utf-8 -*-
"""frozen_bert.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/12minSptATvd2igj-K0W2QO1hDDKi0zrt
"""

#Imports
import torch
import numpy as np

from torch import nn
import torch.nn.functional as F
from torch import optim
import pandas as pd

#! pip install transformers

# x = []
# while True:
#   x.append(np.zeros(100000000))

# Input file paths to the files
train_file_path = '/home/ubuntu/MCDS-Capstone-Project/social_iqa/pre_processed_datasets/train_paper_pandas.csv'
valid_file_path = '/home/ubuntu/MCDS-Capstone-Project/social_iqa/pre_processed_datasets/dev_paper_pandas.csv'

# Read Train Data
train_data = pd.read_csv(train_file_path)
valid_data = pd.read_csv(valid_file_path)

train_data['ans_a'] = train_data['ans_a'].apply(lambda x: x[6:] + " " + x[:6])
valid_data['ans_a'] = valid_data['ans_a'].apply(lambda x: x[6:] + " " + x[:6])

train_data['ans_b'] = train_data['ans_a'].apply(lambda x: x[6:] + " " + x[:6])
valid_data['ans_b'] = valid_data['ans_a'].apply(lambda x: x[6:] + " " + x[:6])

train_data['ans_c'] = train_data['ans_c'].apply(lambda x: x[6:] + " " + x[:6])
valid_data['ans_c'] = valid_data['ans_c'].apply(lambda x: x[6:] + " " + x[:6])


# Use the GPU
# import torch

from transformers import XLNetTokenizer, XLNetModel

tokenizer = XLNetTokenizer.from_pretrained('xlnet-large-cased')
model = XLNetModel.from_pretrained('xlnet-large-cased')
model.eval()
model = model.cuda()


# input_ids = torch.tensor(tokenizer.encode("Hello, my dog is as asd asf asf as", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
# outputs = model(input_ids)
# last_hidden_states = outputs[0]
# last_hidden_states[0][0].shape

# Get features of each of the answers
# We have to get the features for each of the answers so we run roberta on all three options
def get_bert_features(data):

  # Social IQA has 3 options
  ans_list = ['ans_a', 'ans_b', 'ans_c']
  ans_dict = {x:[] for x in ans_list}


  from tqdm import tqdm
  pbar = tqdm(total=len(data))
  
  
  # For all the rows in the dataset
  for i in range(len(data)):
    pbar.update(1)
    elem = data.iloc[i]

    #For each variation in the dataset
    for ans in ans_list:
      sentence = elem[ans]
      # Recieve tokens from Roberta
      input_ids = torch.tensor(tokenizer.encode(sentence, add_special_tokens=True)).unsqueeze(0)  # Batch size 1
      input_ids = input_ids.cuda()
      outputs = model(input_ids)
      last_hidden_states = outputs[0]
      # roberta.extract features provides a representation for the complete sentence.
      # it has a 1024 dimension vector for each word [no.words * 1024]
      # we use the representation of only the first word since that is what is used in the base paper
      # Hence the [0][0]. 
      # You can do what you wish with the features
      ans_dict[ans].append(last_hidden_states[0][-1].cpu().data.numpy())

  return ans_dict

train_dict = get_bert_features(train_data)
valid_dict = get_bert_features(valid_data)

# Segregate Values into each option
train_ans_a = train_dict['ans_a']
train_ans_b = train_dict['ans_b']
train_ans_c = train_dict['ans_c']
train_y = train_data['one_hot_label'].apply(lambda x: x-1)


valid_ans_a = valid_dict['ans_a']
valid_ans_b = valid_dict['ans_b']
valid_ans_c = valid_dict['ans_c']
valid_y = valid_data['one_hot_label'].apply(lambda x: x-1)

cuda = True
from torch.utils import data


# Data Loader
class MyDataset(data.Dataset):
    def __init__(self, ans_a, ans_b, ans_c, y):
        self.ans_a = ans_a
        self.ans_b = ans_b
        self.ans_c = ans_c 
        self.y = y

    def __len__(self):
        return len(self.ans_a)

    def __getitem__(self,index):
        ans_a = self.ans_a[index]
        ans_b = self.ans_b[index]
        ans_c = self.ans_c[index]
        y = self.y[index]
        return ans_a, ans_b, ans_c, y
    
    
num_workers = 2
    
# Training Loader
train_dataset = MyDataset(train_ans_a, train_ans_b, train_ans_c, train_y)

train_loader_args = dict(shuffle=True, batch_size=128, num_workers=num_workers, pin_memory=True) if cuda\
                    else dict(shuffle=True, batch_size=128)
train_loader = data.DataLoader(train_dataset, **train_loader_args)

# Validation Loader
valid_dataset = MyDataset(valid_ans_a, valid_ans_b, valid_ans_c, valid_y)

valid_loader_args = dict(shuffle=True, batch_size=128, num_workers=num_workers, pin_memory=True) if cuda\
                    else dict(shuffle=True, batch_size=128)
valid_loader = data.DataLoader(valid_dataset, **valid_loader_args)

# Simple Multilayer Perceptron
class Model(torch.nn.Module):
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

# Get Model
model = Model()
model = model.cuda()

# Cross Entropy Function
loss_function = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=0,verbose=True, min_lr=1e-6)
# Training Code
from tqdm import tqdm

def compute_loss(x_output, labels):
    np_y = np.array(labels.cpu().data)
    np_x = np.argmax(x_output.cpu().data.numpy(), axis=1)
    result = np.sum(np.array([1 for x, y in zip(np_x, np_y) if x == y]))
    return (result/len(np_y))


for epoch in range(1, 30): ## run the model for 10 epochs
    train_loss, valid_loss = [], []
    model.train()
    accuracy = []
    
    pbar = tqdm(total=len(train_ans_a))

    for batch_id, (ans_a, ans_b, ans_c, target) in enumerate(train_loader):
        ans_a = ans_a.cuda()
        ans_b = ans_b.cuda()
        ans_c = ans_c.cuda()
        target = target.cuda()

        pbar.update(128)
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
    model.eval()
    valid_acc = []
    for batch_id, (ans_a, ans_b, ans_c, target) in enumerate(valid_loader):
        
        ans_a = ans_a.cuda()
        ans_b = ans_b.cuda()
        ans_c = ans_c.cuda()
        target = target.cuda()

        
        output = model(ans_a, ans_b, ans_c)
        loss = loss_function(output, target)
        res_valid = compute_loss(output, target)
        valid_acc.append(res_valid)
        valid_loss.append(loss.item())
    print("Validation Accuracy:", sum(valid_acc)/len(valid_acc))
    print ("Epoch:", epoch, "Training Loss: ", np.mean(train_loss), "Valid Loss: ", np.mean(valid_loss))
    print('-----------------------------------------------------------------------')
    scheduler.step(np.mean(val_loss))





