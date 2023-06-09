from natsort import natsorted
from os import listdir
from os.path import isfile,join
import random
import time
from tqdm import tqdm
import spacy
from typing import List
from transformers import AutoTokenizer
from collections import OrderedDict
import pickle
from torch.utils.data import TensorDataset,Dataset

nlp = spacy.load("en_core_web_sm")


def get_relation_id(rel_type):
    rel_id_dict = {"SuperSub": 0, "SubSuper": 1, "Coref": 2, "NoRel": 3}
    return rel_id_dict[rel_type]


class InputExample(object):
    def __init__(self,doc_content,x_mention,y_mention,z_mention,x_start_char,y_start_char,z_start_char,xy_label,yz_label,xz_label) -> None:
        #self.guid = guid
        self.doc_content = doc_content
        self.x_mention = x_mention
        self.y_mention = y_mention
        self.z_mention = z_mention
        self.x_start_char = x_start_char
        self.y_start_char = y_start_char
        self.z_start_char = z_start_char
        self.xy_label = xy_label
        self.yz_label = yz_label
        self.xz_label = xz_label

class InputFeatures(object):
    """A single set of feature of data"""
    def __init__(self,input_ids,input_mask,segment_ids,label_id,entity=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segement_ids = segment_ids
        self.label_id = label_id
        #maybe have some troubles
        self.entity = entity

class DataProcessor(object):
    """Base class for data converters for sequence classification data sets"""

    def get_train_examples(self):
        """Get a collection of 'InputExample's for the train set"""
        raise NotImplementedError()
    
    def get_dev_examples(self):
        """Get a collection of 'InputExample's for the dev set"""
        raise NotImplementedError()
    
    def get_test_examples(self):
        """Get a collection of 'InputExample's for the test set"""
        raise NotImplementedError
    
    def get_labels(self):
        """Gets the list of labels for this data set"""
        raise [0,1,2,3]

class ESLProcessor(DataProcessor):
    """Processor for the Event Story Line dataset."""

    def __init__(self,symm,downsample,data_dir="./dataset/EventStoryLine/") -> None:
        super().__init__()
        self.data_dir = data_dir
        self.downsample = downsample
        self.symm = symm
        self.all_train_set = []
        self.all_valid_set = []
        self.all_test_set = []

        self.all_valid_cv_set = []
        self.all_test_cv_set = []

        esl_files = natsorted([f for f in listdir(self.data_dir) if isfile(join(self.data_dir,f)) and f[-4:]=="tsvx"])
        train_range,valid_range,test_range = [],[],[]
        keys = list(range(253))
        random.shuffle(keys)
        for (i,key) in enumerate(keys):
            if i<=51:
                test_range.append(key)
            elif i<=102:
                valid_range.append(key)
            else:
                train_range.append(key)
        
        self.esl_train, self.esl_valid, self.esl_test = [],[],[]
        for i,file in enumerate(esl_files):
            if i in train_range:
                self.esl_train.append(file)
            elif i in valid_range:
                self.esl_valid.append(file)
            elif i in test_range:
                self.esl_test.append(file)
            
        # start_time = time.time()
        # print("EventStoryLine train files processing...")
        # for i,file in enumerate(tqdm(train_range)):
        #     data_dict = self.read_tsvx_file(file,self.symm)

    def get_train_examples(self):
        data_dir = self.data_dir
        esl_train = self.esl_train
        all_train_sets = []
        for file_name in esl_train:
            data_dict = self.read_tsvx_file(file_name,self.symm)
            train_set = self.get_esl_train_set(data_dict,self.downsample,self.symm)
            all_train_sets.extend(train_set)
        return all_train_sets
    
    def get_dev_examples(self):
        esl_dev = self.esl_valid
        all_dev_sets = []
        for file_name in esl_dev:
            data_dict = self.read_tsvx_file(file_name,self.symm)
            dev_set = self.get_esl_train_set(data_dict,self.downsample,self.symm)
            all_dev_sets.extend(dev_set)
        return all_dev_sets
    
    def get_test_examples(self):
        esl_test = self.esl_test
        all_test_sets = []
        for file_name in esl_test:
            data_dict = self.read_tsvx_file(file_name,self.symm)
            test_set = self.get_esl_train_set(data_dict,self.downsample,self.symm)
            all_test_sets.extend(test_set)
        return all_test_sets
    
    def get_esl_train_set(self,data_dict,downsample,symm_train):
        train_set = []
        event_dict = data_dict["event_dict"]
        doc_content = data_dict["doc_content"]
        # sntc_dict = data_dict["sentence"]
        relation_dict = data_dict["relation_dict"]
        num_event = len(event_dict)
        keys = list(event_dict.keys())
        for first in range(0,num_event+1):
            for second in range(first+1,num_event):
                for third in range(second+1,num_event):
                    x,y,z = keys[first],keys[second],keys[third]
                    self.append_train_dataset(train_set,downsample,x,y,z,event_dict,doc_content,relation_dict)
                    if symm_train:
                        if relation_dict[(x,y)]["relation"]==0 or relation_dict[(x,y)]["relation"]==1:
                            if (y,x) in relation_dict.keys() and (x,z) in relation_dict.keys() and (y,z) in relation_dict.keys():
                                self.append_train_dataset(train_set,y,x,z,event_dict,doc_content,relation_dict)
                        if relation_dict[(y,z)]["relation"]==0 or relation_dict[(y,z)]["relation"]==1:
                            if (x,z) in relation_dict.keys() and (z,y) in relation_dict.keys() and (x,y) in relation_dict.keys():
                                self.append_train_dataset(train_set,x,z,y,event_dict,doc_content,relation_dict)
                        if relation_dict[(x,z)]["relation"]==0 or relation_dict[(x,z)]["relation"]==1:
                            if (z,y) in relation_dict.keys() and (y,x) in relation_dict.keys() and (z,x) in relation_dict.keys():
                                self.append_train_dataset(train_set,downsample,z,y,x,doc_content,relation_dict)

        return train_set
    
    def read_tsvx_file(self,file,symm):
        data_dict = {}
        data_dict["doc_id"] = file.replace(".tsvx","")
        data_dict["event_dict"] = {}
        data_dict["relation_dict"] = {}
        file_path = self.data_dir + file
        for line in open(file_path,mode="r"):
            line = line.split("\t")
            type = line[0].lower()
            if type == "text":
                data_dict["doc_content"] = line[1]
            elif type == "event":
                event_id, event_word, start_char = int(line[1]), line[2], int(line[4])
                end_char = len(event_word) + start_char - 1
                data_dict["event_dict"][event_id] = {
                "mention": event_word,
                "start_char": start_char,
                "end_char": end_char,
                }
            elif type == "relation":
                event_id1, event_id2, rel_type = int(line[1]), int(line[2]), line[3]
                rel_id = get_relation_id(rel_type)
                data_dict["relation_dict"][(event_id1, event_id2)] = {}
                data_dict["relation_dict"][(event_id1, event_id2)]["relation"] = rel_id
            
                if symm:
                    if line[3]=="SuperSub":
                        self.add_symmertric_data(data_dict,int(line[1]),int(line[2]),"SubSuper")
                    if line[3]=="SubSuper":
                        self.add_symmertric_data(data_dict,int(line[1]),int(line[2]),"SuperSub")
            
            else:
                raise ValueError("File is not in HiEve tsvx format...")
        
        #add additional Noref relation
        event_keys = list(data_dict["event_dict"].keys())
        num_event = len(event_keys)
        for first in range(0,num_event):
            for second in range(first+1,num_event):
                if (event_keys[first],event_keys[second]) not in data_dict["relation_dict"].keys():
                    data_dict["relation_dict"][(event_keys[first],event_keys[second])]={}
                    data_dict["relation_dict"][(event_keys[first],event_keys[second])]["relation"] = get_relation_id("NoRel")
        return data_dict

    #add symmertric data for the event ralation is SuperSub or SubSuper
    def add_symmertric_data(self,data_dict,event_id1,event_id2,rel_type):
        pass

    def append_train_dataset(self,train_set,downsample,x,y,z,event_dict,doc_content,relation_dict):
        x_start_char = event_dict[x]["start_char"]
        y_start_char = event_dict[y]["start_char"]
        z_start_char = event_dict[z]["start_char"]

        x_mention = event_dict[x]["mention"]
        y_mention = event_dict[y]["mention"]
        z_mention = event_dict[z]["mention"]

        #label{"SuperSub": 0, "SubSuper": 1, "Coref": 2, "NoRel": 3}
        xy_label = relation_dict[(x,y)]["relation"] 
        yz_label = relation_dict[(y,z)]["relation"] 
        xz_label = relation_dict[(x,z)]["relation"] 

        #(self,doc_content,x_mention,y_mention,z_mention,x_start_char,y_start_char,z_start_char)
        if xy_label == 3 and yz_label== 3:
            pass
        elif xy_label==3 or yz_label==3 or xz_label==3:
            if random.uniform(0,1) < downsample:
                train_set.append(InputExample(doc_content,x_mention,y_mention,z_mention,x_start_char,y_start_char,z_start_char
                                              ,relation_dict[(x,y)]["relation"],relation_dict[(y,z)]["relation"],relation_dict[(x,z)]["relation"]))
        else:
            train_set.append(InputExample(doc_content,x_mention,y_mention,z_mention,x_start_char,y_start_char,z_start_char
                                          ,relation_dict[(x,y)]["relation"],relation_dict[(y,z)]["relation"],relation_dict[(x,z)]["relation"]))
        

class BartDataset(Dataset):
    def __init__(self, features):
        super().__init__()
        self.features = features
    
    def __getitem__(self, idx):
        return self.features[idx]
    
    def __len__(self, ):
        return len(self.features)

def convert_data_to_feature_for_bart(examples,max_seq_length,tokenizer,args,rel2id):
    """Loads a data file into a list of 'InputBatch's."""

    print("loadding")

    instances = []
    save_file = "./dataset/instances.pkl"

    for (ex_index,example) in enumerate(examples):
        tokens = []
        X_START = "[x_start]"
        X_END = "[x_end]"
        Y_START = "[y_start]"
        Y_END = "[y_end]"
        Z_START = "[z_start]"
        Z_END = "[z_end]"

        if ex_index%10000==0:
            print("Writing example {} of {}".format(ex_index,len(examples)))
        tokens = nlp(example.doc_content)
        x_token_index = len(nlp(example.doc_content[:example.x_start_char]))
        y_token_index = len(nlp(example.doc_content[:example.y_start_char]))
        z_token_index = len(nlp(example.doc_content[:example.z_start_char]))

        append_tokens = []
        for i,token in enumerate(tokens):
            if i == x_token_index:
                append_tokens.append(X_START)
            if i == y_token_index:
                append_tokens.append(Y_START)
            if i == z_token_index:
                append_tokens.append(Z_START)    
            append_tokens.append(str(token))
            if i == x_token_index:
                append_tokens.append(X_END)
            if i == y_token_index:
                append_tokens.append(Y_END)
            if i == z_token_index:
                append_tokens.append(Z_END)
        #append prompt
        prompt = f"{example.x_mention} {tokenizer.mask_token} {example.y_mention} {tokenizer.mask_token} {example.z_mention} {tokenizer.mask_token} {example.x_mention}"
        

        if ex_index == 0:
            input_text = " ".join(append_tokens)
            print(f"input text : {input_text}")
            print(f"prompt : {prompt}" )
            print(f"xy_label : {example.xy_label} yz_label : {example.yz_label} xz_label : {example.xz_label}")
        inputs = tokenizer(
            prompt,
            " ".join(append_tokens),
            truncation = "longest_first",
            max_length = max_seq_length,
            padding = "max_length",
            add_special_tokens = True
        )

        # num_tokens += sum(inputs["attention_mask"])

        if sum(inputs["attention_mask"]) > max_seq_length:
            pass
        
        x = OrderedDict()
        x["input_ids"] = inputs["input_ids"]
        x["attention_mask"] = inputs["attention_mask"]
        x["xy_label"] = example.xy_label
        x["yz_label"] = example.yz_label
        x["xz_label"] = example.xz_label
        x["labels"] = {0:"SuperSub",1:"SubSuper",2:"Coref",3:"NoRef"}
        instances.append(x)
    
    with open(file = save_file,mode="wb") as fw:
        pickle.dump(instances,fw)
    print("Finish save preprocessed data to {}".format(save_file))

    dataset = BartDataset(instances)

    return dataset


def get_dataset(mode,args,tokenizer,processor):
    examples = []
    if mode == "train":
        examples = processor.get_train_examples()
    elif mode == "dev":
        examples = processor.get_dev_examples()
    elif mode == "test":
        examples = processor.get_test_examples()
    else:
        raise Exception("mode must be in choice [train,dev,test]")
    
    dataset = convert_data_to_feature_for_bart(examples,args.max_seq_length,tokenizer,args,processor.get_labels())

    return dataset


processors = {"ESL":ESLProcessor}

if __name__ == "__main__":
    processor = ESLProcessor(True,0.1,"./dataset/EventStoryLine/")
    processor.get_train_examples()
    processor.get_dev_examples()
    test = processor.get_test_examples()
    convert_data_to_feature_for_bart(test,512,AutoTokenizer.from_pretrained("bert-base-uncased"),None,None)
    # for example in test:
    #     print(example.x_mention,example.y_mention,example.z_mention,example.xy_label,example.yz_label,example.xz_label)
    # print(len(test))
    # print("complete!")

