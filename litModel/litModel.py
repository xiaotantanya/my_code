from argparse import ArgumentParser
from logging import debug
import pytorch_lightning as pl
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .base import *
import random


class BertLitModel(BaseLitModel):
    """
    use AutoModelForMaskedLM, and select the output by another layer in the lit model
    """
    def __init__(self,model,args,tokenizer):
        super().__init__(model,args)
        self.tokenizer = tokenizer

        with open(f"{args.data_dir}/rel2id.json","r") as file:
            rel2id = json.load(file)
        
        self.loss_fn = nn.CrossEntropyLoss()
        self.eval_fn = None
        self.best_f1 = 0
        self.t_lambda = args.t_lambda
        
        self.label_st_id = tokenizer("[class0]",add_special_tokens=False)["input_ids"][0]
        self.tokenizer = tokenizer

        # self.__init__label_word()
    
    def forward(self,x):
        return self.model(x)
    
    def training_step(self,batch,batch_idx):
        input_ids,attention_mask,labels,so = batch
        result = self.model(input_ids,attention_mask,return_dict=True,output_hidden_states=True)
        logits = result.logits
        output_embedding = result.hidden_states[-1]
        logits = self.pvp(logits,input_ids)
        ke_loss = self.ke_loss(output_embedding,labels,so,input_ids)
        loss = self.loss_fn(logits,labels)+self.t_lambda*ke_loss
        self.log("Train_loss",loss)
        self.log("Train/ke_loss",ke_loss)

    def get_loss(self,logits,input_ids,labels):
        _, mask_idx = (input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)
        bs = input_ids.shape[0]
        mask_output = logits[torch.arange(bs), mask_idx]
        
        loss = self.loss_fn(mask_output, labels)
        return loss
    
    def validation_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        input_ids, attention_mask, labels, _ = batch
        logits = self.model(input_ids, attention_mask, return_dict=True).logits
        # logits = self.model.roberta(input_ids, attention_mask).last_hidden_state
        # loss = self.loss_fn(logits, labels)
        logits = self.pvp(logits, input_ids)
        loss = self.loss_fn(logits, labels)
        self.log("Eval/loss", loss)
        return {"eval_logits": logits.detach().cpu().numpy(), "eval_labels": labels.detach().cpu().numpy()}
    
    def validation_epoch_end(self, outputs) -> None:
        logits = np.concatenate([o["eval_logits"] for o in outputs])
        labels = np.concatenate([o["eval_labels"] for o in outputs])

        f1 = self.eval_fn(logits, labels)['f1']
        self.log("Eval/f1", f1)
        if f1 > self.best_f1:
            self.best_f1 = f1
        self.log("Eval/best_f1", self.best_f1, prog_bar=True, on_epoch=True)

    def test_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        input_ids, attention_mask, labels, _ = batch
        logits = self.model(input_ids, attention_mask, return_dict=True).logits
        logits = self.pvp(logits, input_ids)
        return {"test_logits": logits.detach().cpu().numpy(), "test_labels": labels.detach().cpu().numpy()}

    def test_epoch_end(self, outputs) -> None:
        logits = np.concatenate([o["test_logits"] for o in outputs])
        labels = np.concatenate([o["test_labels"] for o in outputs])

        f1 = self.eval_fn(logits, labels)['f1']
        self.log("Test/f1", f1)
    
    @staticmethod
    def add_to_argparse(parser):
        BaseLitModel.add_to_argparse(parser)
        parser.add_argument("--t_lambda", type=float, default=0.01, help="")
        parser.add_argument("--t_gamma", type=float, default=0.3, help="")
        return parser
        
    def pvp(self, logits, input_ids):
        # convert the [batch_size, seq_len, vocab_size] => [batch_size, num_labels]
        #! hard coded
        _, mask_idx = (input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)
        bs = input_ids.shape[0]
        mask_output = logits[torch.arange(bs), mask_idx]
        assert mask_idx.shape[0] == bs, "only one mask in sequence!"
        final_output = mask_output[:,self.word2label]
        
        return final_output
        
    def ke_loss(self, logits, labels, so, input_ids):
        subject_embedding = []
        object_embedding = []
        neg_subject_embedding = []
        neg_object_embedding = []
        bsz = logits.shape[0]
        for i in range(bsz):
            subject_embedding.append(torch.mean(logits[i, so[i][0]:so[i][1]], dim=0))
            object_embedding.append(torch.mean(logits[i, so[i][2]:so[i][3]], dim=0))

            # random select the neg samples
            st_sub = random.randint(1, logits[i].shape[0] - 6)
            span_sub = random.randint(1, 5)
            st_obj = random.randint(1, logits[i].shape[0] - 6)
            span_obj = random.randint(1, 5)
            neg_subject_embedding.append(torch.mean(logits[i, st_sub:st_sub+span_sub], dim=0))
            neg_object_embedding.append(torch.mean(logits[i, st_obj:st_obj+span_obj], dim=0))
            
        subject_embedding = torch.stack(subject_embedding)
        object_embedding = torch.stack(object_embedding)
        neg_subject_embedding = torch.stack(neg_subject_embedding)
        neg_object_embedding = torch.stack(neg_object_embedding)
        # trick , the relation ids is concated, 


        _, mask_idx = (input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)
        mask_output = logits[torch.arange(bsz), mask_idx]
        mask_relation_embedding = mask_output
        real_relation_embedding = self.model.get_output_embeddings().weight[labels+self.label_st_id]
        
        d_1 = torch.norm(subject_embedding + mask_relation_embedding - object_embedding, p=2) / bsz
        d_2 = torch.norm(neg_subject_embedding + real_relation_embedding - neg_object_embedding, p=2) / bsz
        f = torch.nn.LogSigmoid()
        loss = -1.*f(self.args.t_gamma - d_1) - f(d_2 - self.args.t_gamma)
        
        return loss

    def configure_optimizers(self):
        no_decay_param = ["bias", "LayerNorm.weight"]

        if not self.args.two_steps: 
            parameters = self.model.named_parameters()
        else:
            # model.bert.embeddings.weight
            parameters = [next(self.model.named_parameters())]
        # only optimize the embedding parameters
        optimizer_group_parameters = [
            {"params": [p for n, p in parameters if not any(nd in n for nd in no_decay_param)], "weight_decay": self.args.weight_decay},
            {"params": [p for n, p in parameters if any(nd in n for nd in no_decay_param)], "weight_decay": 0}
        ]

        
        optimizer = self.optimizer_class(optimizer_group_parameters, lr=self.lr, eps=1e-8)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.num_training_steps * 0.1, num_training_steps=self.num_training_steps)
        return {
            "optimizer": optimizer, 
            "lr_scheduler":{
                'scheduler': scheduler,
                'interval': 'step',  # or 'epoch'
                'frequency': 1,
            }
        }
