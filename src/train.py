from transformers import AutoConfig,AutoTokenizer,AutoModel
import logging
from transformers import BertForMaskedLM
from data_reader import data_loader,read_all_file
import torch
from tqdm import tqdm
logger = logging.getLogger()

def multilabel_categorical_crossentropy(y_pred, y_true):
    y_pred = (1 - 2 * y_true) * y_pred
    y_pred_neg = y_pred - y_true * 1e12
    y_pred_pos = y_pred - (1 - y_true) * 1e12
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
    return (neg_loss + pos_loss).mean()


"""
default
    model_name: bert-base-uncased
    model_class: BertForMaskedLM
"""
model_name_or_path = "bert-base-uncased"
model_class = "BertForMaskedLM"
config = AutoConfig.from_pretrained(model_name_or_path)
model = BertForMaskedLM.from_pretrained(model_name_or_path,config=config)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

event_list = ["[subevent_start]","[subevent_end]","[objevent_start]","[objevent_start]"]
class_list = [f"[class{i}]"for i in range(1,5)]
logger.info(event_list)
logger.info(class_list)


num_added_tokens = tokenizer.add_special_tokens({"additional_special_tokens":event_list})
num_added_tokens = tokenizer.add_special_tokens({"additional_special_tokens":class_list})

train_instances,valid_instances,test_instances = read_all_file("./hievents_v2",tokenizer)
train_dataloader,valid_dataloader,test_dataloader = data_loader(train_instances,valid_instances,test_instances)


model.resize_token_embeddings(len(tokenizer))
tokenizer.save_pretrained("test")

continous_label_word = [a[0] for a in tokenizer([f"[class{i}]" for i in range(1, 5)], add_special_tokens=False)['input_ids']]

no_decay_param = ["bias", "LayerNorm.weight"]
parameters = model.named_parameters()
optimizer_group_parameters = [
            {"params": [p for n, p in parameters if not any(nd in n for nd in no_decay_param)], "weight_decay": 0.001},
            {"params": [p for n, p in parameters if any(nd in n for nd in no_decay_param)], "weight_decay": 0}
        ]
optim = torch.optim.Adam(params=optimizer_group_parameters,lr=5e-5,eps=1e-8)
loss_fn = torch.nn.CrossEntropyLoss()
# model.to("cuda:0")
for input_ids,attention_mask,token_type_ids,labels in tqdm(train_dataloader):
    # input_ids,attention_mask,token_type_ids = input_ids.to("cuda:0"),attention_mask.to("cuda:0"),token_type_ids.to("cuda:0")
    # labels = labels.to("cuda:0")
    result = model(input_ids,attention_mask,token_type_ids,return_dict=True,output_hidden_states=True)
    _,mask_idx = (input_ids==103).nonzero(as_tuple=True)
    bs = input_ids.shape[0]
    mask_output = result.logits[torch.arange(bs),mask_idx]
    final_output = mask_output[:,continous_label_word]
    model.zero_grad()
    loss = loss_fn(final_output,labels)
    loss.backward()
    optim.step()

model.eval()
test_loss = 0
target_num = torch.zeros((1,4))
predict_num = torch.zeros((1,4))
acc_num = torch.zeros((1,4))

with torch.no_grad():
    for input_ids,attention_mask,token_type_ids,labels in tqdm(test_dataloader):
        # input_ids,attention_mask,token_type_ids = input_ids.to("cuda:0"),attention_mask.to("cuda:0"),token_type_ids.to("cuda:0")
        # labels = labels.to("cuda:0")
        result = model(input_ids,attention_mask,token_type_ids,return_dict=True,output_hidden_states=True)
        _,mask_idx = (input_ids==103).nonzero(as_tuple=True)
        bs = input_ids.shape[0]
        mask_output = result.logits[torch.arange(bs),mask_idx]
        final_output = mask_output[:,continous_label_word]
        predict = torch.argmax(torch.softmax(final_output,dim=1),dim=1)
        pre_mask = torch.zeros(final_output.size()).scatter_(1,predict.view(-1,1),1)
        predict_num += pre_mask.sum(0)
        tar_mask = torch.zeros(final_output.size()).scatter_(1,labels.data.view(-1,1),1)
        target_num += tar_mask.sum(0)
        acc_mask = pre_mask*tar_mask
        acc_num += acc_mask.sum(0)
        loss = loss_fn(final_output,predict)
        test_loss += loss.item()
    
    recall = acc_num / target_num
    precision = acc_num / predict_num
    F1 = 2 * recall * precision / (recall + precision)
    sum_f1 = 0
    print(target_num)
    accuracy = 100. * acc_num.sum(1) / target_num.sum(1)
    print(F1)
    print(test_loss)
    print(accuracy)