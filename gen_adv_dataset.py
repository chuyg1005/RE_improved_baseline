import os, json
from argparse import ArgumentParser
from utils import predict, loadModelAndProcessor


def main():
    parser = ArgumentParser()
    parser.add_argument("--data_root", default="./data", type=str)
    parser.add_argument("--dataset", required=True, type=str)
    parser.add_argument("--eval_name", required=True, type=str)
    parser.add_argument("--ckpt_dir", default="./ckpts", type=str)
    parser.add_argument("--model_name", default="roberta-large", type=str)
    parser.add_argument("--seed", type=int, default=0,
                        help="random seed for initialization")
    parser.add_argument("--test_batch_size", default=128, type=int)
    parser.add_argument("--device", default="cuda", type=str)

    args = parser.parse_args()

    eval_file = os.path.join(args.data_root, args.dataset, args.eval_name + ".json")
    eval_data = json.load(open(eval_file, "r"))

    def predictByInputFormat(input_format):
        save_path = os.path.join(args.ckpt_dir, args.dataset, input_format, f"{args.model_name}-{args.seed}")
        model, processor = loadModelAndProcessor(save_path)
        eval_features = processor.read(eval_file)
        keys, preds = predict(model, eval_features, args.test_batch_size, args.device)
        return keys, preds

    keys1, entityPreds = predictByInputFormat("typed_entity_name_punct") # entity-only
    keys2, contextPreds = predictByInputFormat("entity_mask") # context-only

    data1 = []
    data2 = []
    data3 = []
    data4 = []

    for i in range(len(keys1)):
        if entityPreds[i] != keys1[i] and contextPreds[i] == keys1[i]: # 上下文比实体名称更有用
            data1.append(eval_data[i])
        if entityPreds[i] == keys1[i] and contextPreds[i] != keys1[i]: # 实体名称比上下文更有用
            data2.append(eval_data[i])
        if entityPreds[i] != keys1[i] and contextPreds[i] != keys1[i]: # 他们都不做不对，困难样本
            data3.append(eval_data[i])
        if entityPreds[i] == keys1[i] and contextPreds[i] == keys1[i]:
            data4.append(eval_data[i])

    with open(os.path.join(args.data_root, args.dataset, f"{args.eval_name}_data1.json"), "w") as f:
        json.dump(data1, f)
    with open(os.path.join(args.data_root, args.dataset, f"{args.eval_name}_data2.json"), "w") as f:
        json.dump(data2, f)
    with open(os.path.join(args.data_root, args.dataset, f"{args.eval_name}_data3.json"), "w") as f:
        json.dump(data3, f)
    with open(os.path.join(args.data_root, args.dataset, f"{args.eval_name}_data4.json"), "w") as f:
        json.dump(data4, f)

if __name__ == '__main__':
    main()