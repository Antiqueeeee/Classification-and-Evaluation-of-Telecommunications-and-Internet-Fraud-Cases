import os
import json
from transformers import AutoTokenizer
import prettytable as pt
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

class myDataset(Dataset):
    def __init__(self,bert_inputs,bert_labels):
        self.bert_intputs = bert_inputs
        self.bert_labels = bert_labels
    def __len__(self):
        return len(self.bert_intputs)
    def __getitem__(self, item):
        return torch.LongTensor(self.bert_intputs[item]),\
               torch.LongTensor(self.bert_labels[item])
class myTest(Dataset):
    def __init__(self,bert_inputs):
        self.bert_intputs = bert_inputs
    def __len__(self):
        return len(self.bert_intputs)
    def __getitem__(self, item):
        return torch.LongTensor(self.bert_intputs[item])
class Vocabulary:
    def __init__(self):
        self.label2id,self.id2label = dict(),dict()
    def add_label(self,label):
        if label not in self.label2id:
            self.id2label[len(self.label2id)] = label
            self.label2id[label] = len(self.label2id)
        assert label == self.id2label[self.label2id[label]]
    def __len__(self):
        return len(self.label2id)
    def label_to_id(self,label):
        return self.label2id[label]
    def id_to_label(self,id):
        return self.id2label[id]
    def save_Vocabulary(self,save_path):
        label2id_path = os.path.join(save_path,"label2id.json")
        id2label_path = os.path.join(save_path,"id2label.json")
        with open(label2id_path, "w", encoding="utf-8") as f:
            json.dump(self.label2id, f, ensure_ascii=False, indent=2)
        with open(id2label_path, "w", encoding="utf-8") as f:
            json.dump(self.id2label, f, ensure_ascii=False, indent=2)
    def load_Vocabulary(self,save_path):
        def jsonKey2int(x):
            if isinstance(x, dict):
                return {int(k) : v for k, v in x.items()}
            return x
        def jsonValue2int(x):
            if isinstance(x, dict):
                return {k : int(v) for k, v in x.items()}
            return x
        label2id_path = os.path.join(save_path, "label2id.json")
        id2label_path = os.path.join(save_path, "id2label.json")
        with open(label2id_path,"r",encoding="utf-8") as f:
            self.label2id = json.load(f,object_hook=jsonValue2int)
        with open(id2label_path,"r",encoding="utf-8") as f:
            self.id2label = json.load(f,object_hook=jsonKey2int)
def process_bert(data,vocab,tokenizer):
    bert_inputs = list()
    bert_labels = list()
    for instance in tqdm(data):
        case_no = instance["案件编号"]
        case_desc = instance["案情描述"]
        case_label = instance["案件类别"]
        vocab.add_label(case_label)
        tokens = [tokenizer.tokenize(word) for word in case_desc]
        pieces = [piece for pieces in tokens for piece in pieces]
        _bert_inputs = tokenizer.convert_tokens_to_ids(pieces)
        _bert_inputs = [tokenizer.cls_token_id] + _bert_inputs + [tokenizer.sep_token_id]
        _bert_labels = [vocab.label_to_id(case_label)]
        bert_inputs.append(_bert_inputs)
        bert_labels.append(_bert_labels)
    return bert_inputs,bert_labels
def process_test(data,vocab,tokenizer):
    bert_inputs = list()
    for instance in tqdm(data):
        case_no = instance["案件编号"]
        case_desc = instance["案情描述"]
        tokens = [tokenizer.tokenize(word) for word in case_desc]
        pieces = [piece for pieces in tokens for piece in pieces]
        _bert_inputs = tokenizer.convert_tokens_to_ids(pieces)
        _bert_inputs = [tokenizer.cls_token_id] + _bert_inputs + [tokenizer.sep_token_id]
        bert_inputs.append(_bert_inputs)
    return bert_inputs
def collate_fn(data):
    _bert_inputs,_bert_labels = map(list,zip(*data))
    inputs_length = [len(i) for i in _bert_inputs]
    batch_size = len(_bert_inputs)
    bert_inputs = torch.zeros(batch_size,max(inputs_length),dtype=torch.long)
    def fill(data,new_data):
        for j, x in enumerate(data):
            new_data[j, :x.shape[0]] = x
        return new_data
    bert_inputs = fill(_bert_inputs, bert_inputs)
    _bert_labels = torch.LongTensor(_bert_labels)
    return bert_inputs,_bert_labels
def test_collate_fn(data):
    _bert_inputs = list(data)
    inputs_length = [len(i) for i in _bert_inputs]
    batch_size = len(_bert_inputs)
    bert_inputs = torch.zeros(batch_size,max(inputs_length),dtype=torch.long)
    def fill(data,new_data):
        for j, x in enumerate(data):
            new_data[j, :x.shape[0]] = x
        return new_data
    bert_inputs = fill(_bert_inputs, bert_inputs)
    return bert_inputs
def load_data_bert(config):
    save_path =  os.path.join(
        os.path.abspath("."), "data", config.task
    )
    train_data_path = os.path.join(
        save_path, config.train_file
    )
    dev_data_path = os.path.join(
        save_path, config.dev_file
    )
    with open(train_data_path,encoding="utf-8") as f:
        train_data = json.load(f)
    with open(dev_data_path,encoding="utf-8") as f:
        dev_data = json.load(f)
    vocab = Vocabulary()
    tokenizer = AutoTokenizer.from_pretrained(config.bert_name,cache_dir = "./cache")
    table = pt.PrettyTable([config.task,"nums"])
    table.add_row(["train",len(train_data)])
    table.add_row(["dev", len(dev_data)])
    config.logger.info(f"\n{table}")

    train_dataset = myDataset(*process_bert(train_data,vocab,tokenizer))
    dev_dataset = myDataset(*process_bert(dev_data,vocab,tokenizer))
    config.label_num = len(vocab.label2id)
    config.vocab = vocab
    vocab.save_Vocabulary(save_path)
    return (train_dataset,dev_dataset),(train_data,dev_data)

    # 深层Transformer
    # 句子向量
    # Label Sematic

def load_data_test(config):
    test_data_path = os.path.join(
        os.path.abspath("."),"data",config.task,config.test_file
    )
    with open(test_data_path,encoding="utf-8") as f :
        test_data = json.load(f)

    save_path = os.path.join(
        os.path.abspath("."), "data", config.task
    )
    vocab = Vocabulary()
    vocab.load_Vocabulary(save_path)
    config.vocab = vocab
    config.label_num = len(vocab.label2id)
    tokenizer = AutoTokenizer.from_pretrained(config.bert_name, cache_dir="./cache")
    test_dataset = myTest(process_test(test_data,vocab,tokenizer))
    return test_dataset,test_data
