import os
import time
import logging
import json
import torch
import numpy as np
import random

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

class Config:
    def __init__(self,args):
        config_path = os.path.join(os.path.abspath("."),"config",args.task,args.config_file)
        with open(config_path,encoding="utf-8") as f:
            config = json.load(f)
        self.batch_size = config["batch_size"]
        self.bert_learning_rate = config["bert_learning_rate"]
        self.non_bert_learning_rate = config["non_bert_learning_rate"]
        self.weight_decay = config["weight_decay"]
        self.epochs = config["epochs"]
        self.bert_hid_size = config["bert_hid_size"]
        self.loss_type = config["loss_type"]
        self.bert_name = config["bert_name"]

        for k,v in args.__dict__.items():
            if v:
                self.__dict__[k] = v


def get_logger(task):
    if not os.path.exists(os.path.join(os.path.abspath("."),"log")):
        os.mkdir(os.path.join(os.path.abspath("."),"log"))

    if not os.path.exists(os.path.join(os.path.abspath("."), "log",task)):
        os.mkdir(os.path.join(os.path.abspath("."), "log",task))
    path_name = os.path.join(
        os.path.abspath("."),"log",task,time.strftime("%m-%d_%H-%M-%S")
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s",
                                  datefmt='%Y-%m-%d %H:%M:%S')
    file_handler = logging.FileHandler(path_name)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger
