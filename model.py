
from transformers import BertModel
import torch

class classificateWithCls(torch.nn.Module):
    def __init__(self,config):
        super(classificateWithCls, self).__init__()
        self.model_name = "classificateWithCls"
        self.config = config
        self.bert = BertModel.from_pretrained(config.bert_name,output_hidden_states=True)
        self.linear = torch.nn.Linear(in_features=self.config.bert_hid_size,out_features=self.config.label_num)
    def forward(self,bert_inputs):
        bert_out = self.bert(bert_inputs,attention_mask=bert_inputs.ne(0).float())
        sequence_output, cls_output = bert_out[0],bert_out[1]
        outputs = self.linear(cls_output)
        return outputs