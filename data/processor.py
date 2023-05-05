from natsort import natsorted
from os import listdir
from os.path import isfile,join
import random
import time
from tqdm import tqdm

def get_relation_id(rel_type: str):
    rel_id_dict = {"SuperSub": 0, "SubSuper": 1, "Coref": 2, "NoRel": 3}
    return rel_id_dict[rel_type]


class InputExample(object):
    def __init__(self,guid,text_xy,text_xz,text_yz,label_xy,label_xz,label_yz) -> None:
        self.guid = guid
        self.text_xy = text_xy
        self.text_xz = text_xz
        self.text_yz = text_yz
        self.label_xy = label_xy
        self.label_xz = label_xz
        self.label_yz = label_yz

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

    def get_train_examples(self,data_dir):
        """Get a collection of 'InputExample's for the train set"""
        raise NotImplementedError()
    
    def get_dev_examples(self,data_dir):
        """Get a collection of 'InputExample's for the dev set"""
        raise NotImplementedError()
    
    def get_labels(self):
        """Gets the list of labels for this data set"""
        raise NotImplementedError()

class ESLProcessor(DataProcessor):
    """Processor for the Event Story Line dataset."""

    def __init__(self,symm,data_dir="./dataset/EventStoryLine/") -> None:
        super.__init__()
        self.data_dir = data_dir
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
        
        esl_train, esl_valid, esl_test = [],[],[]
        for i,file in enumerate(esl_files):
            if i in train_range:
                esl_train.append(file)
            elif i in valid_range:
                esl_valid.append(file)
            elif i in test_range:
                esl_test.append(file)
            
        start_time = time.time()
        print("EventStoryLine train files processing...")
        for i,file in enumerate(tqdm(train_range)):
            data_dict = self.read_tsvx_file(file,self.symm)



    def get_esl_train_set(self,data_dict,downsample,symm_train):
        train_set = []
        event_dict = data_dict["event_dict"]
        sntc_dict = data_dict["sentence"]
        relation_dict = data_dict["relation_dict"]
        num_event = len(event_dict)

        for x in range(1,num_event+1):
            for y in range(x+1,num_event+1):
                for z in range(y+1,num_event+1):
                    self.append_train_dataset(train_set,downsample,x,y,z,event_dict,sntc_dict,relation_dict)
                    if symm_train:
                        if relation_dict[(x,y)]["relation"]==0 or relation_dict[(x,y)]["relation"]==1:
                            if (y,x) in relation_dict.keys() and (x,z) in relation_dict.keys() and (y,z) in relation_dict.keys():
                                self.append_train_dataset(train_set,y,x,z,event_dict,sntc_dict,relation_dict)
                        if relation_dict[(y,z)]["relation"]==0 or relation_dict[(y,z)]["relation"]==1:
                            if (x,z) in relation_dict.keys() and (z,y) in relation_dict.keys() and (x,y) in relation_dict.keys():
                                self.append_train_dataset(train_set,x,z,y,event_dict,sntc_dict,relation_dict)
                        if relation_dict[(x,z)]["relation"]==0 or relation_dict[(x,z)]["relation"]==1:
                            if (z,y) in relation_dict.keys() and (y,x) in relation_dict.keys() and (z,x) in relation_dict.keys():
                                self.append_train_dataset(train_set,downsample,z,y,x,event_dict,sntc_dict,relation_dict)

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
                    self.add_symmertric_data(data_dict,int(line[1]),int(line(2)),"SubSuper")
                if line[3]=="SubSuper":
                    self.add_symmertric_data(data_dict,int(line[1]),int(line[2]),"SuperSub")
            
            else:
                raise ValueError("File is not in HiEve tsvx format...")
            
        return data_dict

    #add symmertric data for the event ralation is SuperSub or SubSuper
    def add_symmertric_data(self,data_dict,event_id1,event_id2,rel_type):
        pass

    def append_train_dataset(self,train_set,downsample,x,y,z,event_dict,sntc_dict,relation_dict):
        pass

