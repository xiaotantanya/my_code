from typing import Union,Any,Dict,List
import xml.etree.ElementTree as ET
from nltk import sent_tokenize
from transformers import BertTokenizer
import spacy
from natsort import natsorted
from pathlib import Path
from os import listdir
from os.path import isfile, join
import json
import os
import pickle
import time
from torch.utils.data import DataLoader
from dataset import EventDataset
nlp = spacy.load("en_core_web_sm")

def data_loader(train_instances,valid_instances,test_instances):
    train_dataset = EventDataset(train_instances)
    valid_dataset = EventDataset(valid_instances)
    test_dataset = EventDataset(test_instances)
    train_dataloader = DataLoader(train_dataset,batch_size=32,shuffle=True,num_workers=2)
    valid_dataloader = DataLoader(valid_dataset,batch_size=32,shuffle=True,num_workers=2)
    test_dataloader = DataLoader(test_dataset,batch_size=32,shuffle=True,num_workers=2)
    return train_dataloader,valid_dataloader,test_dataloader

def read_all_file(path:str,tokenizer:Any)->List[Any]:
    train_instances = []
    valid_instances = []
    test_instances = []
    if os.path.exists("train_instances.pickle") and os.path.exists("valid_instances.pickle") and os.path.exists("test_instances.pickle"):
        with open("train_instances.pickle","rb") as f:
            train_instances=pickle.load(f)
        with open("valid_instances.pickle","rb") as f:
            valid_instances=pickle.load(f)
        with open("test_instances.pickle","rb") as f:
            test_instances=pickle.load(f)
        return train_instances,valid_instances,test_instances
    all_files = natsorted([f for f in listdir(path) if isfile(join(path,f)) and f[-3:]=="xml"])
    all_train_set, all_valid_set, all_test_set = [], [], []
    train_range, valid_range, test_range = [], [], []

    with open(path+"/sorted_dict.json") as f:
        sorted_dict = json.load(f)
    i = 0
    for (key, value) in sorted_dict.items():
        i += 1
        key = int(key)
        if i <= 20:
            test_range.append(key)
        elif i <= 40:
            valid_range.append(key)
        else:
            train_range.append(key)
        
    for i, file in enumerate(all_files):
        if i in train_range:
            all_train_set.append(file)
        elif i in valid_range:
           all_valid_set.append(file)
        elif i in test_range:
            all_test_set.append(file)
    

    for file in all_train_set:
        train_instances.extend(read_one_file(join(path,file),tokenizer=tokenizer))
    for file in all_valid_set:
        valid_instances.extend(read_one_file(join(path,file),tokenizer=tokenizer))
    for file in all_test_set:
        test_instances.extend(read_one_file(join(path,file),tokenizer=tokenizer))
    with open("train_instances.pickle","wb") as f:
        pickle.dump(train_instances,f)
    with open("valid_instances.pickle","wb") as f:
        pickle.dump(valid_instances,f)
    with open("test_instances.pickle","wb") as f:
        pickle.dump(test_instances,f)
    return train_instances,valid_instances,test_instances


#读取xml文件
def read_one_file(file:str,tokenizer:Any)->List[Any]:
    part_instances = []
    tree = ET.parse(file)
    root = tree.getroot()
    doc_content=root[0].text
    event_dict={}
    for item in root[1]:
        event_id,event_word,start_char=int(item[0].text),item[2].text,int(item[3].text)
        end_char = len(event_word)+start_char-1
        event_dict[event_id]={
            "mention":event_word,
            "start_char":start_char,
            "end_char":end_char
        }
    for item in root[2]:
        event_id1=int(item[0].text)
        event_id2=int(item[1].text)
        instance = []
        inputs = append_label(doc_content=doc_content,first_position_start=event_dict[event_id1]["start_char"],
                              second_position_start=event_dict[event_id2]["start_char"],
                              tokenizer=tokenizer)
        label = item[2].text
        instance.append(inputs)
        instance.append(label)
        part_instances.append(instance)
    return part_instances

def append_label(doc_content:str,first_position_start:int,second_position_start:int,tokenizer,first_label:str="e1",second_label:str="e2",max_seq_length:int=512)->Any:
    tokens = nlp(doc_content)
    first_token_index = len(nlp(doc_content[:first_position_start]))
    # print(first_token_index)
    second_token_index = len(nlp(doc_content[:second_position_start]))
    # print(second_token_index)

    SUBEVENT_START = "[subevent_start]"
    SUBEVENT_END = "[subevent_end]"
    OBJEVENT_START = "[objevent_start]"
    OBJEVENT_END = "[objevent_end]"
    
    append_tokens=[]
    for i,token in enumerate(tokens):
        if i==first_token_index:
            append_tokens.append(SUBEVENT_START)
        if i==second_token_index:
            append_tokens.append(OBJEVENT_START)
        append_tokens.append(str(token))
        if i==first_token_index:
            append_tokens.append(SUBEVENT_END)
        if i==second_token_index:
            append_tokens.append(OBJEVENT_END)
    #append prompt
    prompt = f"{tokens[first_token_index]} {tokenizer.mask_token} {tokens[second_token_index]}"
    inputs = tokenizer(
                prompt,
                " ".join(append_tokens),
                truncation="longest_first",
                max_length=max_seq_length,
                padding="max_length",
                add_special_tokens=True
            )
    return inputs



if __name__ =="__main__":
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased",use_fast=True)
    # part_instance=read_one_file("./data_example/article-1126.xml",tokenizer=tokenizer)
    # print(part_instance)
    # # print(data_dict)
    start = time.time()
    read_all_file("./hievents_v2",tokenizer)
    end = time.time()
    print(end-start)
