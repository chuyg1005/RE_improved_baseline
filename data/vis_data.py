import json
import random
import argparse
from googletrans import Translator


def translate(sentence):
    """translate sentence from english to zh-cn"""
    translator = Translator()
    return translator.translate(sentence, dest='zh-cn').text


def main(args):
    random.seed(args.seed)
    vis_path = args.vis_path
    vis_data = json.load(open(vis_path, 'r', encoding='utf-8'))
    vis_data = random.sample(vis_data, args.vis_num)
    for i, item in enumerate(vis_data):
        print('-' * 50)
        en_sent = ' '.join(item['token'])
        print('en_sent: ', en_sent)
        if args.trans:
            zh_sent = translate(en_sent)
            print('zh_sent: ', zh_sent)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--vis_path", default="./data/test.json", type=str)
    parser.add_argument("--vis_num", default=10, type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trans", action="store_true")
    args = parser.parse_args()
    main(args)
