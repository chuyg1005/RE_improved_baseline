import os
import json
import random
from argparse import ArgumentParser
import copy
from tqdm import tqdm

def gen_entity_dict(data):
    print("generating entity dict...")
    entity_dict = {}
    for item in data:
        tokens = item["token"]
        ss, se = item["subj_start"], item["subj_end"]
        os, oe = item["obj_start"], item["obj_end"]
        subj_type, obj_type = item["subj_type"], item["obj_type"]
        subj_span, obj_span = tokens[ss:se+1], tokens[os:oe+1]
        subj, obj = " ".join(subj_span), " ".join(obj_span)

        if subj_type not in entity_dict:
            entity_dict[subj_type] = set()
        if obj_type not in entity_dict:
            entity_dict[obj_type] = set()

        entity_dict[subj_type].add(subj)
        entity_dict[obj_type].add(obj)

    for key in entity_dict.keys():
        entity_dict[key] = list(entity_dict[key])

    return entity_dict

def substitute_item_with_new_entities(item, new_subj, new_obj):
    new_item = copy.deepcopy(item)
    new_subj_span = new_subj.split()
    new_obj_span = new_obj.split()
    ss, se = item["subj_start"], item["subj_end"]
    os, oe = item["obj_start"], item["obj_end"]

    tokens = item["token"]
    new_tokens = []
    new_ss = new_se = 0
    new_os = new_oe = 0

    if ss < os:
        new_tokens.extend(tokens[:ss])
        new_ss = len(new_tokens)
        new_tokens.extend(new_subj_span)
        new_se = len(new_tokens) - 1
        new_tokens.extend(tokens[se+1:os])
        new_os = len(new_tokens)
        new_tokens.extend(new_obj_span)
        new_oe = len(new_tokens) - 1
        new_tokens.extend(tokens[oe+1:])
    else:
        new_tokens.extend(tokens[:os])
        new_os = len(new_tokens)
        new_tokens.extend(new_obj_span)
        new_oe = len(new_tokens) - 1
        new_tokens.extend(tokens[oe+1:ss])
        new_ss = len(new_tokens)
        new_tokens.extend(new_subj_span)
        new_se = len(new_tokens) - 1
        new_tokens.extend(tokens[se+1:])

    new_item["token"] = new_tokens
    new_item["subj_start"] = new_ss
    new_item["subj_end"] = new_se
    new_item["obj_start"] = new_os
    new_item["obj_end"] = new_oe

    return new_item

def gen_aug_dataset(filedir, k):
    filepath = os.path.join(filedir, "train.json")
    if not os.path.exists(filepath):
        print(f"{filepath} not exists.")
        assert 0
    with open(filepath, "r") as f:
        data = json.load(f)

    entity_dict = gen_entity_dict(data)

    print("generating augmented-dataset by entity-switch...")
    aug_data = []
    for item in tqdm(data):
        aug_item = [item]
        subj_type = item["subj_type"]
        obj_type = item["obj_type"]
        for _ in range(k):
            new_subj = random.choice(entity_dict[subj_type])
            new_obj = random.choice(entity_dict[obj_type])
            new_item = substitute_item_with_new_entities(item, new_subj, new_obj)
            aug_item.append(new_item)
        aug_data.append(aug_item)

    with open(os.path.join(filedir, f"train-aug-{k}.json"), "w") as f:
        json.dump(aug_data, f)

def main():
    parser = ArgumentParser()
    parser.add_argument("--data_root", default="./data", type=str)
    parser.add_argument("--dataset", required=True, type=str)
    parser.add_argument("--k", default=3, type=int)
    parser.add_argument("--seed", default=0, type=int)
    args = parser.parse_args()

    random.seed(args.seed)

    filedir = os.path.join(args.data_root, args.dataset)
    gen_aug_dataset(filedir, args.k)

if __name__ == '__main__':
    main()