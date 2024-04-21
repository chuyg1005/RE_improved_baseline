import os, json
from argparse import ArgumentParser
from utils import predict, loadModelAndProcessor, generate_md5_hash, compute_f1
import numpy as np


def main(args):
    if args.dataset == 'tacrev':
        train_dataset = 'tacred'
    else:
        train_dataset = args.dataset

    save_path = os.path.join(args.ckpt_dir, train_dataset, args.mode, args.split,
                             generate_md5_hash(args.input_format, args.split, args.model_name, args.mode,
                                               args.train_name, str(args.seed)))
    model, processor = loadModelAndProcessor(save_path, args.device)

    eval_co_file = os.path.join(args.data_root, args.dataset, args.test_split, args.eval_name + "-co.json")
    eval_eo_file = os.path.join(args.data_root, args.dataset, args.test_split, args.eval_name + "-eo.json")
    eval_file = os.path.join(args.data_root, args.dataset, args.test_split, args.eval_name + ".json")

    eval_co_features = processor.read(eval_co_file)
    eval_eo_features = processor.read(eval_eo_file)
    eval_features = processor.read(eval_file)

    if args.version == 'version1':
        print("use version1")
        keys, preds_co = predict(model, eval_co_features, args.test_batch_size, args.device)
        keys, preds_eo = predict(model, eval_eo_features, args.test_batch_size, args.device)
        keys, preds = predict(model, eval_features, args.test_batch_size, args.device)

        label2id = processor.LABEL_TO_ID
        # id2label = {v: k for k, v in label2id.items()}

        # grouped by label_ids
        keys = np.array(keys, dtype=np.int64)
        preds_co = np.array(preds_co, dtype=np.int64)
        preds_eo = np.array(preds_eo, dtype=np.int64)
        preds = np.array(preds, dtype=np.int64)

        label_dict = {}
        for label in label2id:
            label_id = label2id[label]

            indices = np.where(keys == label_id)[0]

            keys_part = keys[indices]
            preds_co_part = preds_co[indices]
            preds_eo_part = preds_eo[indices]
            preds_part = preds[indices]

            co_acc = np.sum(preds_co_part == keys_part) / len(keys_part)
            eo_acc = np.sum(preds_eo_part == keys_part) / len(keys_part)
            acc = np.sum(preds_part == keys_part) / len(keys_part)

            if eo_acc > co_acc:
                label_dict[label] = {
                    "co_acc": co_acc,
                    "eo_acc": eo_acc,
                    "acc": acc
                }

        with open(os.path.join("./label_dict.json"), 'w') as f:
            json.dump(label_dict, f, indent=4)
        print(json.dumps(label_dict, indent=4))
    elif args.version == 'version2':
        print("use version2")
        keys, preds_eo, probs_eo = predict(model, eval_features, args.test_batch_size, args.device, withProbs=True)
        keys = np.array(keys, dtype=np.int64)
        preds_eo = np.array(preds_eo, dtype=np.int64)
        probs_eo = np.array(probs_eo)
        preds_eo_indices = probs_eo.argsort(axis=1)[:, ::-1]

        keys = keys.reshape(-1, 1)
        for i in range(1, args.k + 1):
            preds_eo_topk = preds_eo_indices[:, :i]  # [n, k]
            equal_matrix = preds_eo_topk == keys  # [n, k]
            # print(equal_matrix)
            preds = np.any(equal_matrix, axis=1)  # [n]
            acc = np.sum(preds) / len(preds)
            print(f"top-{i} acc: {acc}")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--data_root", default="./data", type=str)
    parser.add_argument("--dataset", required=True, type=str)
    parser.add_argument("--eval_name", default="test", type=str)
    parser.add_argument("--split", default="origin", type=str)
    parser.add_argument("--test_split", default="origin", type=str)
    parser.add_argument("--ckpt_dir", default="./ckpts", type=str)
    parser.add_argument("--model_name", default="bert-base-cased", type=str)
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument("--train_name", default="train4debias")
    parser.add_argument("--input_format", default="typed_entity_marker_punct", type=str,
                        help="in [entity_mask, entity_marker, entity_marker_punct, typed_entity_marker, typed_entity_marker_punct]")
    parser.add_argument("--mode", default="default", type=str)
    parser.add_argument("--test_batch_size", default=128, type=int)
    parser.add_argument("--device", default="cuda", type=str)

    parser.add_argument("--version", default="version1", choices=["version1", "version2"])
    parser.add_argument("--k", default=8, type=int)

    args = parser.parse_args()
    main(args)
