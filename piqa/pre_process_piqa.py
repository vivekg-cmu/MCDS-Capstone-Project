
# coding: utf-8

# In[1]:


# Imports
import pandas as pd
import numpy as np
import json


# In[6]:


# File paths
train_json_path = "/home/pratik/Desktop/capstone/MCDS-Capstone-Project/piqa/dataset/train.jsonl"
dev_json_path = "/home/pratik/Desktop/capstone/MCDS-Capstone-Project/piqa/dataset/valid.jsonl"
train_labels_path = "/home/pratik/Desktop/capstone/MCDS-Capstone-Project/piqa/dataset/train-labels.lst"
dev_labels_path = "/home/pratik/Desktop/capstone/MCDS-Capstone-Project/piqa/dataset/valid-labels.lst"


# In[7]:


# Read json data
def read_json(file_path):
    result = None
    with open(file_path, 'r') as fh:
        result = [json.loads(x) for x in fh.readlines()]
    return result
json_data_train = read_json(train_json_path)
json_data_dev = read_json(dev_json_path)


# In[8]:


def read_labels(file_path):
    result = None
    with open(file_path, 'r') as fh:
        result = [int(x) for x in fh.readlines()]
    return result

train_y = read_labels(train_labels_path)
dev_y = read_labels(dev_labels_path)


# In[16]:


# Json Format
# {'answerA': 'like attending',
#   'answerB': 'like staying home',
#   'context': 'Cameron decided to have a barbecue and gathered her friends together.',
#   'question': 'How would Others feel as a result?'}

def convert_json_to_pandas(json_data):
    
    json_dict = {"goal" : [],
    "sol1" : [],
    "sol2" : [],
}
    for json_line in json_data:
        json_dict["goal"].append(json_line['goal'])
        json_dict["sol1"].append(json_line['sol1'])
        json_dict["sol2"].append(json_line['sol2'])
    
    return pd.DataFrame.from_dict(json_dict)

train_pandas_x = convert_json_to_pandas(json_data_train)
dev_pandas_x = convert_json_to_pandas(json_data_dev)


# In[17]:


def merge_x_and_labels(pandas_x, labels):
    pandas_x['labels'] = labels
    return pandas_x

train_pandas = merge_x_and_labels(train_pandas_x, train_y)
dev_pandas = merge_x_and_labels(dev_pandas_x, dev_y)


# In[24]:


# Code to be used like trained in base paper

def generate_binary_table(data_pandas):
    json_dict =         {
        "goal" : [],
        "sol1" : [],
        "sol2" : [],
        "one_hot_label": []
        }
    
    CLS = "[CLS]"
    SEP = "[SEP]"
    UNUSED = "[UNUSED]"
    
    
    ans_list = ['sol1', 'sol2']
    
    for i in range(len(data_pandas)):
        elem = data_pandas.iloc[i]
        goal = elem['goal']
        json_dict['goal'].append(goal)
        json_dict['one_hot_label'].append(elem['labels'])
        for j in range(len(ans_list)):
            answer = elem[ans_list[j]]            
            result = CLS + " " + goal + " " + SEP + " " + answer + SEP
            json_dict[ans_list[j]].append(result)
            
    return pd.DataFrame.from_dict(json_dict)

train_paper_pandas = generate_binary_table(train_pandas)
dev_paper_pandas = generate_binary_table(dev_pandas)


# In[28]:


pd.DataFrame.to_csv(train_paper_pandas, '/pre_processed_datasets/train_paper_pandas.csv')
pd.DataFrame.to_csv(dev_paper_pandas, '/pre_processed_datasets/dev_paper_pandas.csv')

