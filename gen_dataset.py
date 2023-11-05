import os
import json
import random
from argparse import ArgumentParser
import copy
from tqdm import tqdm
from utils import loadModelAndProcessor, predict


def gen_entity_dict(data):
    print("generating entity dict...")
    entity_dict = {}
    for item in data:
        tokens = item["token"]
        ss, se = item["subj_start"], item["subj_end"]
        os, oe = item["obj_start"], item["obj_end"]
        subj_type, obj_type = item["subj_type"], item["obj_type"]
        subj_span, obj_span = tokens[ss:se + 1], tokens[os:oe + 1]
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
        new_tokens.extend(tokens[se + 1:os])
        new_os = len(new_tokens)
        new_tokens.extend(new_obj_span)
        new_oe = len(new_tokens) - 1
        new_tokens.extend(tokens[oe + 1:])
    else:
        new_tokens.extend(tokens[:os])
        new_os = len(new_tokens)
        new_tokens.extend(new_obj_span)
        new_oe = len(new_tokens) - 1
        new_tokens.extend(tokens[oe + 1:ss])
        new_ss = len(new_tokens)
        new_tokens.extend(new_subj_span)
        new_se = len(new_tokens) - 1
        new_tokens.extend(tokens[se + 1:])

    new_item["token"] = new_tokens
    new_item["subj_start"] = new_ss
    new_item["subj_end"] = new_se
    new_item["obj_start"] = new_os
    new_item["obj_end"] = new_oe

    return new_item


def gen_dfl_dataset(filedir):
    filepath = os.path.join(filedir, "train.json")
    if not os.path.exists(filepath):
        print(f"{filepath} not exists.")
        assert 0
    with open(filepath, "r") as f:
        data = json.load(f)
    aug_data = []
    for item in tqdm(data):
        new_item = copy.deepcopy(item)

        tokens = item["token"]
        ss, se = item["subj_start"], item["subj_end"]
        _os, oe = item["obj_start"], item["obj_end"]
        subj_span, obj_span = tokens[ss:se + 1], tokens[_os:oe + 1]

        new_item["token"] = subj_span + obj_span  # 删除and
        new_item["subj_start"] = 0
        new_item["subj_end"] = len(subj_span) - 1
        new_item["obj_start"] = len(subj_span)
        new_item["obj_end"] = len(subj_span) + len(obj_span) - 1

        # 丢弃实体类型信息
        # new_item["subj_type"] = "SUBJ_TYPE"
        # new_item["obj_type"] = "OBJ_TYPE"
        #
        aug_data.append([item, new_item])

    with open(os.path.join(filedir, f"train-dfl.json"), "w") as f:
        json.dump(aug_data, f)


def gen_adv_dataset(filedir, k):
    filepath = os.path.join(filedir, "train.json")
    with open(filepath, "r") as f:
        data = json.load(f)

    ckpt_dir = "./ckpts"
    dataset = os.path.basename(filedir)
    input_format = "typed_entity_name_punct"
    model_name = "bert-base-cased"
    device = "cuda"

    save_path = os.path.join(ckpt_dir, dataset, input_format, f"{model_name}-default-train-0")
    model, processor = loadModelAndProcessor(save_path, device)

    entity_dict = gen_entity_dict(data)

    aug_data = []
    for item in tqdm(data):
        aug_item = [item]
        subj_type = item['subj_type']
        obj_type = item['obj_type']
        candidates = []
        for _ in range(5 * k):
            new_subj = random.choice(entity_dict[subj_type])
            new_obj = random.choice(entity_dict[obj_type])
            new_item = substitute_item_with_new_entities(item, new_subj, new_obj)
            candidates.append(new_item)
        features = processor.encode(candidates, show_bar=False)
        _, _, probs = predict(model, features, len(candidates), device, withProbs=True)
        # 按照Probs降序排列，获取indices
        indices = sorted(range(len(probs)), key=lambda i: probs[i], reverse=False)
        candidates = [candidates[idx] for idx in indices]
        aug_item += candidates[:k]
        aug_data.append(aug_item)

    with open(os.path.join(filedir, f"train-adv-{k}.json"), "w") as f:
        json.dump(aug_data, f)


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
    parser.add_argument("--k", default=5, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--mode", default='adv', type=str)
    args = parser.parse_args()

    random.seed(args.seed)

    filedir = os.path.join(args.data_root, args.dataset)
    # gen_aug_dataset(filedir, args.k)
    if args.mode == 'adv':
        gen_adv_dataset(filedir, args.k)
    else:
        gen_aug_dataset(filedir, args.k)
        # gen_dfl_dataset(filedir)


if __name__ == '__main__':
    main()
