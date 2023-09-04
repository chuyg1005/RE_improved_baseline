import torch
import random
import pickle
import numpy as np
from torch.utils.data import DataLoader
import json, os
from model import REModel


def get_f1(key, prediction):
    correct_by_relation = ((key == prediction) & (prediction != 0)).astype(np.int32).sum()
    guessed_by_relation = (prediction != 0).astype(np.int32).sum()
    gold_by_relation = (key != 0).astype(np.int32).sum()

    prec_micro = 1.0
    if guessed_by_relation > 0:
        prec_micro = float(correct_by_relation) / float(guessed_by_relation)
    recall_micro = 1.0
    if gold_by_relation > 0:
        recall_micro = float(correct_by_relation) / float(gold_by_relation)
    f1_micro = 0.0
    if prec_micro + recall_micro > 0.0:
        f1_micro = 2.0 * prec_micro * recall_micro / (prec_micro + recall_micro)
    return prec_micro, recall_micro, f1_micro


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0 and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

def predict(model, features, test_batch_size, device):
    dataloader = DataLoader(features, batch_size=test_batch_size, collate_fn=collate_fn, drop_last=False)
    keys, preds = [], []
    for i_b, batch in enumerate(dataloader):
        model.eval()

        inputs = {'input_ids': batch[0].to(device),
                  'attention_mask': batch[1].to(device),
                  'ss': batch[3].to(device),
                  'os': batch[4].to(device),
                  }
        keys += batch[2].tolist()
        with torch.no_grad():
            logit = model(**inputs)[0]
            pred = torch.argmax(logit, dim=-1)
        preds += pred.tolist()

    return keys, preds

def evaluate(model, features, test_batch_size, device):
    keys, preds = predict(model, features, test_batch_size, device)
    keys = np.array(keys, dtype=np.int64)
    preds = np.array(preds, dtype=np.int64)
    _, _, max_f1 = get_f1(keys, preds)

    return max_f1 * 100

def collate_fn(batch):
    max_len = max([len(f["input_ids"]) for f in batch])
    input_ids = [f["input_ids"] + [0] * (max_len - len(f["input_ids"])) for f in batch]
    input_mask = [[1.0] * len(f["input_ids"]) + [0.0] * (max_len - len(f["input_ids"])) for f in batch]
    labels = [f["labels"] for f in batch]
    ss = [f["ss"] for f in batch]
    os = [f["os"] for f in batch]
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    input_mask = torch.tensor(input_mask, dtype=torch.float)
    labels = torch.tensor(labels, dtype=torch.long)
    ss = torch.tensor(ss, dtype=torch.long)
    os = torch.tensor(os, dtype=torch.long)
    output = (input_ids, input_mask, labels, ss, os)
    return output

def saveModelStateDict(model, filepath):
    torch.save(model.state_dict(), filepath)

def loadModelStateDict(filepath):
    return torch.load(filepath)

def loadModelAndProcessor(save_path):
    __args = load4File(os.path.join(save_path, "args.pkl"))
    config = load4File(os.path.join(save_path, "config.pkl"))
    processor = load4File(os.path.join(save_path, "processor.pkl"))

    # load model
    model = REModel(__args, config)
    model.to(0)
    model.load_state_dict(loadModelStateDict(os.path.join(save_path, "best.pth")))
    return model, processor

def dump2File(obj, filepath):
    with open(filepath, "wb") as f:
        pickle.dump(obj, f)


def load4File(filepath):
    with open(filepath, "rb") as f:
        return pickle.load(f)


def genMockData():
    filepath = os.path.join('./data', 'tacred', 'train.json')
    data_dir = os.path.join('./data', 'tacmock')
    os.makedirs(data_dir, exist_ok=True)
    data = json.load(open(filepath, "r"))

    print(f"generate mock data from {filepath} for test.")
    with open(os.path.join(data_dir, "train.json"), "w") as f: json.dump(data[:10240], f)
    with open(os.path.join(data_dir, "dev.json"), "w") as f: json.dump(data[:64], f)
    with open(os.path.join(data_dir, "test.json"), "w") as f: json.dump(data[:64], f)

if __name__ == '__main__':
    genMockData()
