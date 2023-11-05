import os, json
from argparse import ArgumentParser
from utils import predict, loadModelAndProcessor


def main():
    parser = ArgumentParser()
    parser.add_argument("--data_root", default="./data", type=str)
    parser.add_argument("--dataset", required=True, type=str)
    parser.add_argument("--eval_name", required=True, type=str)
    parser.add_argument("--mode", default="default", type=str)
    parser.add_argument("--ckpt_dir", default="./ckpts", type=str)
    parser.add_argument("--model_name", default="bert-base-cased", type=str)
    parser.add_argument("--seed", type=int, default=0,
                        help="random seed for initialization")
    parser.add_argument("--test_batch_size", default=128, type=int)
    parser.add_argument("--device", default="cuda", type=str)

    args = parser.parse_args()

    eval_file = os.path.join(args.data_root, args.dataset, args.eval_name + ".json")
    eval_data = json.load(open(eval_file, "r"))

    def predictByInputFormat(input_format):
        save_path = os.path.join(args.ckpt_dir, args.dataset, input_format,
                                 f"{args.model_name}-{args.mode}-train-{args.seed}")
        model, processor = loadModelAndProcessor(save_path, args.device)
        eval_features = processor.read(eval_file)
        keys, preds = predict(model, eval_features, args.test_batch_size, args.device)
        return keys, preds

    keys1, entityPreds = predictByInputFormat("typed_entity_name_punct")  # entity-only
    keys2, contextPreds = predictByInputFormat("entity_mask")  # context-only

    # data1 = []
    # data2 = []
    # data3 = []
    # data4 = []
    challenge_data = []

    for i in range(len(keys1)):
        if entityPreds[i] != keys1[i]:
            # if entityPreds[i] != keys1[i] and contextPreds[i] == keys1[i]:
            challenge_data.append(eval_data[i])
        # if entityPreds[i] != keys1[i] and contextPreds[i] == keys1[i]: # 上下文正确但是实体名称错误
        #     data1.append(eval_data[i])
        # if entityPreds[i] == keys1[i] and contextPreds[i] != keys1[i]: # 实体名称正确但是上下文错误
        #     data2.append(eval_data[i])
        # if entityPreds[i] != keys1[i] and contextPreds[i] != keys1[i]: # 二者都做不对
        #     data3.append(eval_data[i])
        # if entityPreds[i] == keys1[i] and contextPreds[i] == keys1[i]: # 二者都能做对, 简单样本
        #     data4.append(eval_data[i])

    with open(os.path.join(args.data_root, args.dataset, f"{args.eval_name}_challenge.json"), "w") as f:
        json.dump(challenge_data, f)
    # with open(os.path.join(args.data_root, args.dataset, f"{args.eval_name}_data1.json"), "w") as f:
    #     json.dump(data1, f)
    # with open(os.path.join(args.data_root, args.dataset, f"{args.eval_name}_data2.json"), "w") as f:
    #     json.dump(data2, f)
    # with open(os.path.join(args.data_root, args.dataset, f"{args.eval_name}_data3.json"), "w") as f:
    #     json.dump(data3, f)
    # with open(os.path.join(args.data_root, args.dataset, f"{args.eval_name}_data4.json"), "w") as f:
    #     json.dump(data4, f)


if __name__ == '__main__':
    main()
