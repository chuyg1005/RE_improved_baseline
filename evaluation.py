import os
from argparse import ArgumentParser
from utils import evaluate, loadModelAndProcessor
import numpy as np, json


def main():
    parser = ArgumentParser()
    parser.add_argument("--data_root", default="./data", type=str)
    parser.add_argument("--dataset", required=True, type=str)
    parser.add_argument("--eval_name", required=True, type=str)
    parser.add_argument("--ckpt_dir", default="./ckpts", type=str)
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument("--mode", default="default", type=str)
    parser.add_argument("--test_batch_size", default=128, type=int)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--save_predict", action="store_true", help="save prediction results")

    args = parser.parse_args()

    model_name = f'{args.mode}-{args.seed}'
    model_path = os.path.join(args.ckpt_dir, args.dataset, model_name)
    model, processor = loadModelAndProcessor(model_path, args.device)
    label2id = processor.LABEL_TO_ID

    eval_file = os.path.join(args.data_root, args.dataset, args.eval_name + ".json")
    save_dir = os.path.join(args.data_root, 'predictions', 'ibre', args.dataset, model_name)
    eval_features = processor.read(eval_file)

    score, result = evaluate(model, eval_features, args.test_batch_size, args.device, return_result=True)
    print(f"evaluate best model on {args.eval_name}, F1-score: {score:.2f}")

    # 保存label2id和预测结果
    if args.save_predict:
        os.makedirs(save_dir, exist_ok=True)
        label2id_path = os.path.join(save_dir, "label2id.json")
        pred_path = os.path.join(save_dir, f"{args.eval_name}.txt")
        with open(label2id_path, "w") as f:
            json.dump(label2id, f)
        with open(pred_path, "w") as f:
            print(result.shape)
            np.savetxt(f, result, fmt="%.4f", delimiter=",")
        print(f"save prediction results to {pred_path}")


if __name__ == '__main__':
    main()
