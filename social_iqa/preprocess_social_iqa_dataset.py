#PLEASE CHANGE THE TRAIN DEV FILE PATHS AND ALSO PATHS AT THE END OF THE FILE WHERE THE DATA HAS TO BE SAVED
# coding: utf-8

# In[1]:


# Imports
import pandas as pd
import numpy as np
import json


# In[4]:


# File paths
train_json_path = "/socialiqa-train-dev/train.jsonl"
dev_json_path = "/socialiqa-train-dev/dev.jsonl"
train_labels_path = "/socialiqa-train-dev/train-labels.lst"
dev_labels_path = "/socialiqa-train-dev/dev-labels.lst"


# In[5]:


# Read json data
def read_json(file_path):
    result = None
    with open(file_path, 'r') as fh:
        result = [json.loads(x) for x in fh.readlines()]
    return result
json_data_train = read_json(train_json_path)
json_data_dev = read_json(dev_json_path)


# In[6]:


def read_labels(file_path):
    result = None
    with open(file_path, 'r') as fh:
        result = [int(x) for x in fh.readlines()]
    return result

train_y = read_labels(train_labels_path)
dev_y = read_labels(dev_labels_path)


# In[7]:


# Json Format
# {'answerA': 'like attending',
#   'answerB': 'like staying home',
#   'answerC': 'a good friend to have',
#   'context': 'Cameron decided to have a barbecue and gathered her friends together.',
#   'question': 'How would Others feel as a result?'}

def convert_json_to_pandas(json_data):
    
    json_dict = {"ans_a" : [],
    "ans_b" : [],
    "ans_c" : [],
    "context" : [],
    "question" : []
}
    for json_line in json_data:
        json_dict["ans_a"].append(json_line['answerA'])
        json_dict["ans_b"].append(json_line['answerB'])
        json_dict["ans_c"].append(json_line['answerC'])
        json_dict["context"].append(json_line['context'])
        json_dict["question"].append(json_line['question'])
    
    return pd.DataFrame.from_dict(json_dict)

train_pandas_x = convert_json_to_pandas(json_data_train)
dev_pandas_x = convert_json_to_pandas(json_data_dev)


# In[8]:


def merge_x_and_labels(pandas_x, labels):
    pandas_x['labels'] = labels
    return pandas_x

train_pandas = merge_x_and_labels(train_pandas_x, train_y)
dev_pandas = merge_x_and_labels(dev_pandas_x, dev_y)


# In[9]:


# Code to be used like trained in base paper

def generate_binary_table(data_pandas):
    json_dict =         {
        "ans_a" : [],
        "ans_b" : [],
        "ans_c" : [],
        "one_hot_label": []
        }
    
    CLS = "[CLS]"
    SEP = "[SEP]"
    UNUSED = "[UNUSED]"
    
    
    ans_list = ['ans_a', 'ans_b', 'ans_c']
    
    for i in range(len(data_pandas)):
        elem = data_pandas.iloc[i]
        context = elem['context']
        question = elem['question']
        json_dict['one_hot_label'].append(elem['labels'])
        for j in range(len(ans_list)):
            answer = elem[ans_list[j]]            
            result = CLS + " " + context + " " + UNUSED + " " + question + " " + SEP + " " + answer + SEP
            json_dict[ans_list[j]].append(result)
            
    
    return pd.DataFrame.from_dict(json_dict)

train_paper_pandas = generate_binary_table(train_pandas)
dev_paper_pandas = generate_binary_table(dev_pandas)


# In[ ]:


pd.DataFrame.to_csv(train_paper_pandas, '/pre_processed_datasets/train_paper_pandas2.csv')
pd.DataFrame.to_csv(dev_paper_pandas, '/pre_processed_datasets/dev_paper_pandas2.csv')

