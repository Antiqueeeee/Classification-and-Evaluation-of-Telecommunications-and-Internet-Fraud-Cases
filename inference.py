from data_loader import load_data_test,test_collate_fn
from torch.utils.data import DataLoader
from model import classificateWithCls
import argparse
import torch
import os
import json
from tqdm import tqdm
class Predictor:
    def __init__(self,model,config):
        self.model = model
        self.config = config
    def load(self,path):
        self.model.load_state_dict(
            torch.load(path)
        )
    def predict(self,data_loader,original_data):
        result = list()
        index = 0
        with torch.no_grad():
            for data_batch in tqdm(data_loader):
                batch = original_data[ index : index + self.config.batch_size]
                # data_batch = [data.to(self.config.device) for data in data_batch]
                data_batch = data_batch.to(self.config.device)
                outputs = self.model(data_batch)
                for item,pred_label in zip(batch,outputs):
                    text_desc = item["案情描述"]
                    pred_label = torch.argmax(pred_label,-1)
                    pred_label = self.config.vocab.id_to_label(pred_label.cpu().item())
                    result.append({
                        "案件描述":text_desc,
                        "案件类别":pred_label
                    })
                index += self.config.batch_size
        save_path = os.path.join(
            os.path.abspath("."), "data", config.task
        )
        with open(os.path.join(save_path,"model_predicted.json"),"w",encoding="utf-8") as f:
            json.dump(result,f,indent=2,ensure_ascii=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="deceitClassification")
    parser.add_argument("--test_file", type=str, default="test.json")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--bert_hid_size", type=int, default=768)

    parser.add_argument('--save_path', type=str, default="outputs")
    parser.add_argument('--bert_name', type=str, default=r"E:\MyPython\Pre-train-Model\bert-base-chinese")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    config = args
    datasets,ori_data = load_data_test(config)
    test_dataloader = DataLoader(
        dataset = datasets,
        batch_size = config.batch_size,
        collate_fn = test_collate_fn,
    )
    model = classificateWithCls(config).to(config.device)
    predictor = Predictor(model,config)
    predictor.load(
        os.path.join(
            os.path.abspath("."), config.save_path, config.task, predictor.model.model_name + ".pt"
        )
    )
    predictor.predict(test_dataloader, ori_data)