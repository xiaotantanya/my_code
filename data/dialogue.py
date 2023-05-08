from .base_data_module import BaseDataModule
from .processor import get_dataset,processors
from transformers import AutoTokenizer

from dataclasses import dataclass
from torch.utils.data import DataLoader

import random
import warnings
from typing import Any,Dict,List,Tuple,Optional,Union

from transformers.file_utils import PaddingStrategy
from transformers.models.bert import BertTokenizer, BertTokenizerFast
from transformers.tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase

class ESL(BaseDataModule):
    def __init__(self,args)->None:
        super().__init__(args)
        self.processor = processors[self.args.task_name](self.args.data_dir,self.args.use_prompt)
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_name_or_path)

        self.num_labels = len(self.processor.get_labels())
        
        event_list = ["[subject_start]","subject_end","object_start","object_end"]
        class_list = [f"[class{i}]" for i in range(0,self.num_labels)]

        num_add_tokens = self.tokenizer.add_special_tokens({"additional_special_tokens":event_list})
        num_add_tokens = self.tokenizer.add_special_tokens({"additional_special_tokens":class_list})

        so_list = ["[sub]","[obj]"]
        num_add_toens = self.tokenizer.add_special_tokens({"additional_special_tokens":so_list})

        prompt_tokens = [f"[T{i}]" for i in range(1,6)]
        self.tokenizer.add_special_tokens({'additional_special_tokens': prompt_tokens})
    
    def setup(self,stage=None):
        self.data_train = get_dataset("train",self.args,self.tokenizer,self.processor)
        self.data_val = get_dataset("dev",self.args,self.tokenizer,self.processor)
        self.data_test = get_dataset("test",self.args,self.tokenizer,self.processor)
    
    def prepare_data(self):
        pass

    def get_tokenizer(self):
        return self.tokenizer
    
    @staticmethod
    def add_to_argparse(parser):
        BaseDataModule.add_to_argparse(parser)
        parser.add_argument("--task_name",type=str,default="normal",help="[normal,reloss,ptune]")
        parser.add_argument("--model_name_or_path",type=str,default="/home/tanwen/bert-base-uncased",help="Number of examples to operate on per forward step.")
        parser.add_argument("--max_seq_length",type=int,default=512,help="the max sequence word length")
        parser.add_argument("--ptune-k",type=int,default=7,help="experiment to research few-shot")
    
    