import torch.nn

from data_loader import load_data_bert, collate_fn

import argparse
from utils import get_logger,Config,setup_seed
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AdamW
import prettytable as pt
import numpy as np
from model import classificateWithCls
from sklearn.metrics import precision_score,recall_score,f1_score
import os
class Trainer:
    def __init__(self,model,config):
        self.model = model
        criterion = {
            "ce" : torch.nn.CrossEntropyLoss()
        }
        self.criterion = criterion[config.loss_type]
        bert_params = set(self.model.bert.parameters())
        other_params = list(set(self.model.parameters()) - bert_params)
        no_decay = ["bias","LayerNorm.weight"]
        params = [
            {
                "params" : [p for n,p in model.bert.named_parameters() if not any(nd in n for nd in no_decay)],
                "lr" : config.bert_learning_rate,
                "weight_decay" : config.weight_decay
            },
            {
                "params" : [p for n,p in model.bert.named_parameters() if any(nd in n for nd in no_decay)],
                "lr" : config.bert_learning_rate,
                "weight_decay" : 0.0
            },
            {
                "params" : other_params,
                "lr" : config.non_bert_learning_rate,
                "weight_decay" : config.weight_decay
            }
        ]

        # self.optimizer = AdamW(params,lr=config.non_bert_learning_rate,weight_decay=config.weight_decay)
        self.optimizer = AdamW(params)
    def train(self,epoch_index,data_loader):
        self.model.train()
        loss_list = list()
        for i, data_batch in tqdm(enumerate(data_loader)):
            data_batch = [data.to(config.device) for data in data_batch]
            bert_inputs,bert_labels = data_batch
            outputs = self.model(bert_inputs)
            loss = self.criterion(outputs,bert_labels)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            loss_list.append(loss.cpu().item())
            # break
        table = pt.PrettyTable([f"Training","Loss"])
        table.add_row([f"{epoch_index} epoch",f"{np.mean(loss_list):.4f}"])
        logger.info(f"\n{table}")
    def eval(self,epoch_index,data_loader):
        self.model.eval()
        original_labels = list()
        pred_labels = list()
        with torch.no_grad():
            for i,data_batch in tqdm(enumerate(data_loader)):
                data_batch = [data.to(config.device) for data in data_batch]
                bert_inputs,bert_labels = data_batch
                outputs = self.model(bert_inputs)
                for original_label,pred_label,bert_input in zip(bert_labels,outputs,bert_inputs):
                    pred_label = torch.argmax(pred_label,-1)
                    original_label = config.vocab.id_to_label(original_label.cpu().item())
                    pred_label = config.vocab.id_to_label(pred_label.cpu().item())
                    original_labels.append(original_label)
                    pred_labels.append(pred_label)
                # break

        p = precision_score(original_labels,pred_labels,average="macro")
        r = recall_score(original_labels,pred_labels,average="macro")
        f1 = f1_score(original_labels,pred_labels,average="macro")
        table = pt.PrettyTable([f"Dev","F1","Precision","Recall"])
        table.add_row([f"{epoch_index} epoch"] + [f"{x:3.4f}" for x in [f1,p,r]])
        logger.info(f"\n{table}")
        return f1
    def save(self):

        torch.save(
            self.model.state_dict(),
            os.path.join(
                os.path.abspath("."),config.save_path,config.task,self.model.model_name + ".pt"
            )
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="deceitClassification")
    parser.add_argument("--config_file", type=str, default="bert-base-chinese.json")
    parser.add_argument("--train_file", type=str, default="sample_train.json")
    parser.add_argument("--dev_file", type=str, default="sample_dev.json")
    # parser.add_argument("--train_file", type=str, default="train_data.json")
    # parser.add_argument("--dev_file", type=str, default="dev_data.json")
    parser.add_argument('--save_path', type=str, default="outputs")
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()
    config = Config(args)
    os.makedirs(
        os.path.join(os.path.abspath("."), config.save_path,config.task)
    )
    config.save_path = os.path.join(os.path.abspath("."),config.save_path)
    logger = get_logger(config.task)
    config.logger = logger
    setup_seed(2023)
    datasets, ori_data = load_data_bert(config)
    train_dataloader, dev_loader = (DataLoader(dataset=dataset,
                                               batch_size=config.batch_size,
                                               collate_fn=collate_fn,
                                               shuffle=i == 0,
                                               drop_last=i == 0
                                               )
                                    for i, dataset in enumerate(datasets)
                                    )
    model = classificateWithCls(config).to(config.device)
    trainer = Trainer(model,config)

    best_f1 = 0
    for i in range(config.epochs):
        trainer.train(i,train_dataloader)
        f1 = trainer.eval(i,dev_loader)
        if f1 > best_f1:
            best_f1 = f1
            trainer.save()
    logger.info(f"Best dev f1:{best_f1:3.4f}")
