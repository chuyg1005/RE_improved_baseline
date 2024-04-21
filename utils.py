import torch
import random
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer
import json, os
from model import REModel
from types import SimpleNamespace


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
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def predict(model, features, test_batch_size, device, withProbs=False):
    dataloader = DataLoader(features, batch_size=test_batch_size, collate_fn=default_collate_fn, drop_last=False,
                            shuffle=False)
    keys, preds = [], []
    probs = []
    for i_b, batch in enumerate(dataloader):
        model.eval()

        inputs = {'input_ids': batch[0].to(device),
                  'attention_mask': batch[1].to(device),
                  'ss': batch[3].to(device),
                  'os': batch[4].to(device),
                  }
        labels = batch[2].to(device)
        with torch.no_grad():
            logit = model(**inputs)  # predict results
            prob = torch.softmax(logit, dim=1)
            # 获取prob
            _, pred = torch.max(prob, dim=1)
            # label_prob = torch.gather(prob, dim=1, index=labels.unsqueeze(1)).squeeze()
        keys += labels.tolist()
        preds += pred.tolist()
        probs += prob.tolist()  # 2d的概率

    if not withProbs:
        return keys, preds
    else:
        return keys, preds, probs


def compute_f1(keys, preds):
    keys = np.array(keys, dtype=np.int64)
    preds = np.array(preds, dtype=np.int64)
    _, _, max_f1 = get_f1(keys, preds)
    return max_f1


def evaluate(model, features, test_batch_size, device, return_result=False):
    keys, preds, probs = predict(model, features, test_batch_size, device, withProbs=True)
    max_f1 = compute_f1(keys, preds)

    keys = np.array(keys, dtype=np.float32).reshape(-1, 1)
    results = np.hstack([probs, keys])

    if return_result:
        return max_f1 * 100, results
    else:
        return max_f1 * 100


def get_collate_fn(mode="default"):
    """ignore mode and tokenizer, return [batch_org, batch_aug, batch_eo]"""
    return get_MixDebias_collate_fn(mode)


def default_collate_fn(batch):
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


def get_MixDebias_collate_fn(mode):
    print("get MixDebias collate_fn with mode: ", mode)

    def MixDebias_collate_fn(batch):
        # 如果不是list，则直接用default_collate_fn
        batch_org = [f[0] for f in batch]
        batch_eo = [f[1] for f in batch]
        batch_co = [f[2] for f in batch]
        batch_aug = [random.choice(f[3:]) for f in batch]
        if mode == 'default' or mode == 'Focal':
            batch = batch_org
        elif mode == 'EntityOnly':
            batch = batch_eo
        elif mode == 'EntityMask':
            batch = batch_co
        elif mode == 'RDrop':
            batch = batch_org + batch_org
        elif mode == 'DFocal' or mode == 'PoE':
            batch = batch_org + batch_eo
        elif mode == 'Debias':
            batch = batch_org + batch_co
        elif mode == 'RDataAug' or mode == 'DataAug':
            batch = batch_org + batch_aug
        elif mode == 'MixDebias':
            batch = batch_org + batch_co + batch_aug
        else:
            assert 0
        return default_collate_fn(batch)

    return MixDebias_collate_fn


def saveModelStateDict(model, filepath):
    torch.save(model.state_dict(), filepath)


def buildModel(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    config = AutoConfig.from_pretrained(
        args.model_name,
        num_labels=args.num_class,
    )
    config.gradient_checkpointing = True
    config.num_tokens = len(tokenizer)
    model = REModel(args, config)

    return model, tokenizer


def loadModel(save_path, device):
    args = load4File(os.path.join(save_path, "args.json"))
    args = SimpleNamespace(**args)
    model, tokenizer = buildModel(args)
    model_path = os.path.join(save_path, "best.pth")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    return model, tokenizer, args


def dump2File(obj, filepath):
    with open(filepath, "w") as f:
        json.dump(obj, f)


def load4File(filepath):
    with open(filepath, "r") as f:
        return json.load(f)


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
