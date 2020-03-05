#Imports
import torch
import numpy as np

from torch import nn
import torch.nn.functional as F
from torch import optim
import pandas as pd

# Input file paths to the files
train_file_path = '/pre_processed_datasets/train_paper_pandas.csv'
valid_file_path = '/pre_processed_datasets/dev_paper_pandas.csv'


print('Start Loading Roberta')
# Load Roberta Model n evaluation mode
roberta = torch.hub.load('pytorch/fairseq', 'roberta.large')
roberta.eval()
print('Done Loading Roberta')

# Read Train Data
train_data = pd.read_csv(train_file_path)
valid_data = pd.read_csv(valid_file_path)

#Train Head
print(train_data.head(2))

# Move roberta model to GPU
roberta = roberta.cuda()

# Get features of each of the solutions
# We have to get the features for each of the answers so we run roberta on all three options
def get_roberta_features(data):
  # Piqa has 2 solutions
  ans_list = ['sol1', 'sol2']
  ans_dict = {x:[] for x in ans_list}


  from tqdm import tqdm_notebook
  pbar = tqdm_notebook(total=len(data))
  
  
  # For all the rows in the dataset
  for i in range(len(data)):
    pbar.update(1)
    elem = data.iloc[i]

    #For each variation in the dataset
    for ans in ans_list:
      sentence = elem[ans]


      # Recieve tokens from Roberta
      tokens = roberta.encode(sentence)

      # roberta.extract features provides a representation for the complete sentence.
      # it has a 1024 dimension vector for each word [no.words * 1024]
      # we use the representation of only the first word since that is what is used in the base paper
      # Hence the [0][0]. 
      # You can do what you wish with the features
      features = roberta.extract_features(tokens)[0][0].cpu().data.numpy()
      ans_dict[ans].append(features)

  return ans_dict
# Extracted Features in form of a dict

train_dict = get_roberta_features(train_data)
valid_dict = get_roberta_features(valid_data)

# Segregate Values into each option
train_sol1 = train_dict['sol1']
train_sol2 = train_dict['sol2']
train_y = train_data['one_hot_label']


valid_sol1 = valid_dict['sol1']
valid_sol2 = valid_dict['sol2']
valid_y = valid_data['one_hot_label']

cuda = True
from torch.utils import data


# Data Loader
class MyDataset(data.Dataset):
    def __init__(self, sol1, sol2, y):
        self.sol1 = sol1
        self.sol2 = sol2
        self.y = y

    def __len__(self):
        return len(self.sol1)

    def __getitem__(self,index):
        sol1 = self.sol1[index]
        sol2 = self.sol2[index]
        y = self.y[index]
        return sol1, sol2, y
    
    
num_workers = 8
    
# Training Loader
train_dataset = MyDataset(train_sol1, train_sol2, train_y)

train_loader_args = dict(shuffle=True, batch_size=32, num_workers=num_workers, pin_memory=False) if cuda\
                    else dict(shuffle=True, batch_size=32)
train_loader = data.DataLoader(train_dataset, **train_loader_args)

# Validation Loader
valid_dataset = MyDataset(valid_sol1, valid_sol2, valid_y)

valid_loader_args = dict(shuffle=True, batch_size=32, num_workers=num_workers, pin_memory=False) if cuda\
                    else dict(shuffle=True, batch_size=32)
valid_loader = data.DataLoader(valid_dataset, **valid_loader_args)


for elem in train_loader:
  print(elem[0])
  break

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

  
  def forward(self, ans_a, ans_b):
        
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


        output = torch.cat((ans_a, ans_b), dim=1)
      

        return output

# Get Model
model = Model()
model = model.cuda()

# Cross Entropy Function
loss_function = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())



# Training Code
from tqdm import tqdm_notebook

def compute_loss(x_output, labels):
    np_y = np.array(labels.cpu().data)
    np_x = np.argmax(x_output.cpu().data.numpy(), axis=1)
    result = np.sum(np.array([1 for x, y in zip(np_x, np_y) if x == y]))
    return (result/len(np_y))


for epoch in range(1, 100): ## run the model for 10 epochs
    train_loss, valid_loss = [], []
    model.train()
    accuracy = []
    
    pbar = tqdm_notebook(total=len(train_sol1))

    for batch_id, (ans_a, ans_b, target) in enumerate(train_loader):
        ans_a = ans_a.cuda()
        ans_b = ans_b.cuda()
        target = target.cuda()

        pbar.update(128)
        optimizer.zero_grad()
        
        ## 1. forward propagation
        output = model(ans_a, ans_b)
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
    for batch_id, (ans_a, ans_b, target) in enumerate(valid_loader):
        
        ans_a = ans_a.cuda()
        ans_b = ans_b.cuda()
        target = target.cuda()

        output = model(ans_a, ans_b)
        loss = loss_function(output, target)
        res_valid = compute_loss(output, target)
        valid_acc.append(res_valid)
        valid_loss.append(loss.item())
    print("Validation Accuracy:", sum(valid_acc)/len(valid_acc))
    print ("Epoch:", epoch, "Training Loss: ", np.mean(train_loss), "Valid Loss: ", np.mean(valid_loss))
    print('-----------------------------------------------------------------------')

