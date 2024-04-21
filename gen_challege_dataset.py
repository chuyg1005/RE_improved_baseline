import os
from argparse import ArgumentParser
from utils import predict, loadModelAndProcessor, set_seed
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import numpy as np
import json
import copy
import random
from collections import defaultdict
from tqdm import tqdm
from utils import extract_entity_only


class Generator:
    def __init__(self, save_path, entity_dict_path, device='cuda', batch_size=128, version="v1"):
        self.lm_name = 'gpt2'
        self.device = device
        self.version = version
        # 使用gpt2作为语言模型计算句子的perplexity
        if self.version == 'v2':
            self.lm_tokenizer = GPT2Tokenizer.from_pretrained(os.path.join("/home/nfs02/chuyg/models", self.lm_name))
            self.lm_model = GPT2LMHeadModel.from_pretrained(os.path.join("/home/nfs02/chuyg/models", self.lm_name)).to(
                device)
            self.lm_model.eval()

            # 使用mention-only的模型计算概率
            self.eo_model, self.processor = loadModelAndProcessor(save_path, device)
            self.batch_size = batch_size
            self.entity_dict = json.load(open(entity_dict_path))
        elif self.version == 'v1':  # 随机选择类型相同的实体进行替换
            self.entity_dict = json.load(open(entity_dict_path))
        elif self.version == 'v0':  # 对模型进行counterfactual analysis，找出无法正确预测的hard样本
            self.model, self.processor = loadModelAndProcessor(save_path, device)

    def generate(self, item):
        if self.version == 'v1':
            return self._generate_v1(item)
        elif self.version == 'v2':
            return self._generate_v2(item)
        elif self.version == 'v0':
            return self._generate_v0(item)

    def _generate_v1(self, item):
        subj_type = item['subj_type']
        obj_type = item['obj_type']
        new_subj_span = random.choice(self.entity_dict[subj_type]).split()
        new_obj_span = random.choice(self.entity_dict[obj_type]).split()
        new_item = self.substitute_entity(item, new_subj_span, new_obj_span)
        return new_item

    def _generate_v2(self, item):
        new_items = []
        tokens = item['token']
        subj_type = item['subj_type']
        obj_type = item['obj_type']
        subj_start, subj_end = item['subj_start'], item['subj_end']
        obj_start, obj_end = item['obj_start'], item['obj_end']
        for i in range(self.batch_size):
            if subj_type in self.entity_dict:
                new_subj_span = random.choice(self.entity_dict[subj_type]).split()
            else:
                new_subj_span = tokens[subj_start:subj_end + 1]
            if obj_type in self.entity_dict:
                new_obj_span = random.choice(self.entity_dict[obj_type]).split()
            else:
                new_obj_span = tokens[obj_start:obj_end + 1]
            # print(new_subj_span, new_obj_span)
            new_item = self.substitute_entity(item, new_subj_span, new_obj_span)
            new_items.append(new_item)

        biases = self.computeBias(new_items)
        perplexities = self.computePerplexityByGPT2(new_items)

        # normalize biases and perplexities
        biases = np.array(biases)
        indices = np.argsort(biases)[:10]
        perplexities = np.array(perplexities)
        # 寻找perplexities中最小的
        perplexities = perplexities[indices]
        index = np.argmin(perplexities)
        index = indices[index]

        new_item = new_items[index]

        return new_item

    def computeBias(self, items):
        """compute bias using the entity-only model"""
        new_items = []
        for item in items:
            new_item = extract_entity_only(item)
            new_items.append(new_item)

        features = self.processor.encode(new_items, show_bar=False)
        _, _, probs = predict(self.eo_model, features, len(new_items), self.device, True)
        labels = [feature['labels'] for feature in features]
        label_probs = []
        for i in range(len(labels)):
            label_probs.append(probs[i][labels[i]])

        return label_probs

    def _generate_v0(self, item):
        entity_item = extract_entity_only(item)
        items = [entity_item]
        features = self.processor.encode(items, show_bar=False)
        keys, preds = predict(self.model, features, 1, self.device, False)
        if keys[0] == preds[0]:  # 预测正确，则删除
            return None
        else:
            return item

    @staticmethod
    def substitute_entity(item, new_subj_span, new_obj_span):
        new_item = copy.deepcopy(item)
        tokens = item['token']
        ss, se, os, oe = item['subj_start'], item['subj_end'], item['obj_start'], item['obj_end']

        if ss < os:
            new_item['token'] = tokens[:ss] + new_subj_span + tokens[se + 1:os] + new_obj_span + tokens[oe + 1:]
            new_item['subj_start'] = ss
            new_item['subj_end'] = ss + len(new_subj_span) - 1
            new_item['obj_start'] = ss + len(new_subj_span) + os - se - 1
            new_item['obj_end'] = new_item['obj_start'] + len(new_obj_span) - 1
        else:
            new_item['token'] = tokens[:os] + new_obj_span + tokens[oe + 1:ss] + new_subj_span + tokens[se + 1:]
            new_item['obj_start'] = os
            new_item['obj_end'] = os + len(new_obj_span) - 1
            new_item['subj_start'] = os + len(new_obj_span) + ss - oe - 1
            new_item['subj_end'] = new_item['subj_start'] + len(new_subj_span) - 1

        return new_item

    def computePerplexityByGPT2(self, items):
        perplexities = []
        sentences = [' '.join(item['token']) for item in items]
        for sentence in sentences:
            perplexities.append(self.__computePerplexity(sentence))
        return perplexities

    def __computePerplexity(self, sentence):
        tokenize_input = self.lm_tokenizer.tokenize(sentence)
        tensor_input = torch.tensor([self.lm_tokenizer.convert_tokens_to_ids(tokenize_input)]).to(self.device)
        loss = self.lm_model(tensor_input, labels=tensor_input)
        return np.exp(loss[0].item())


def main(args):
    train_dataset = args.dataset
    if args.dataset == "tacrev":
        train_dataset = "tacred"
    generateEntityDict(args.data_root, args.dataset, args.version, args.eval_name)
    save_path = os.path.join(args.ckpt_dir, train_dataset, args.mode, args.split,
                             generate_md5_hash(args.input_format, args.split, args.model_name, args.mode,
                                               args.train_name, str(args.seed)))
    entity_dict_path = os.path.join(args.data_root, args.dataset, f"entity-dict-{args.version}.json")
    test_data_path = os.path.join(args.data_root, args.dataset, "origin", f"{args.eval_name}.json")
    test_challenge_data_path = os.path.join(args.data_root, args.dataset, "origin",
                                            f"{args.eval_name}_challenge_{args.version}.json")
    generator = Generator(save_path, entity_dict_path, device=args.device, batch_size=args.batch_size,
                          version=args.version)

    test_data = json.load(open(test_data_path))
    test_challenge_data = []
    for item in tqdm(test_data):
        new_item = generator.generate(item)
        if new_item is not None:
            test_challenge_data.append(new_item)

    json.dump(test_challenge_data, open(test_challenge_data_path, 'w'))


def generateEntityDict(data_root, dataset, version, eval_name):
    entity_dict_wiki = json.load(
        open(os.path.join(data_root, "entity-dict-wiki.json"))) if version == 'v2' else {}
    data_path = os.path.join(data_root, dataset, "origin", f"{eval_name}.json")
    entity_dict = defaultdict(set)
    data = json.load(open(data_path))
    for item in data:
        entity_dict[item['subj_type']].add(' '.join(item['token'][item['subj_start']:item['subj_end'] + 1]))
        entity_dict[item['obj_type']].add(' '.join(item['token'][item['obj_start']:item['obj_end'] + 1]))
    # merge with entity_dict_wiki
    for entity_type, entities in entity_dict_wiki.items():
        if entity_type in entity_dict:
            entity_dict[entity_type] = entity_dict[entity_type].union(entities)
    new_entity_dict = {}
    for k, v in entity_dict.items():
        new_entity_dict[k] = list(v)
    json.dump(new_entity_dict,
              open(os.path.join(data_root, dataset, f"entity-dict-{version}.json"), 'w'))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--data_root", default="./data", type=str)
    parser.add_argument("--dataset", default='tacred', type=str, choices=['tacred', 'tacrev', 'retacred'])
    parser.add_argument("--split", default="origin", type=str)
    parser.add_argument("--ckpt_dir", default="./ckpts", type=str)
    parser.add_argument("--model_name", default="bert-base-cased", type=str)
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument("--eval_name", type=str, default="test")
    parser.add_argument("--train_name", default="train4debias")
    parser.add_argument("--input_format", default="typed_entity_marker_punct", type=str,
                        help="in [entity_mask, entity_marker, entity_marker_punct, typed_entity_marker, typed_entity_marker_punct]")
    parser.add_argument("--mode", default="EntityOnly", type=str)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--device_id", default=0, type=int)
    parser.add_argument("--version", default="v2", type=str, help="in [v1, v2, v0]", choices=["v1", "v2", "v0"])
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_id)
    set_seed(args)  # 固定随机数
    main(args)
    # main(args)
