import os
from argparse import ArgumentParser
from utils import  evaluate, loadModelAndProcessor


def main():
    parser = ArgumentParser()
    parser.add_argument("--data_root", default="./data", type=str)
    parser.add_argument("--dataset", required=True, type=str)
    parser.add_argument("--eval_name", required=True, type=str)
    parser.add_argument("--ckpt_dir", default="./ckpts", type=str)
    parser.add_argument("--model_name", default="roberta-large", type=str)
    parser.add_argument("--seed", type=int, default=0,
                        help="random seed for initialization")
    parser.add_argument("--input_format", default="typed_entity_marker_punct", type=str,
                        help="in [entity_mask, entity_marker, entity_marker_punct, typed_entity_marker, typed_entity_marker_punct]")
    parser.add_argument("--test_batch_size", default=128, type=int)
    parser.add_argument("--device", default="cuda", type=str)

    args = parser.parse_args()
    save_path = os.path.join(args.ckpt_dir, args.dataset, args.input_format,f"{args.model_name}-{args.seed}")
    model, processor = loadModelAndProcessor(save_path)

    eval_file = os.path.join(args.data_root, args.dataset, args.eval_name + ".json")
    eval_features = processor.read(eval_file)

    score = evaluate(model, eval_features, args.test_batch_size, args.device)
    print(f"evaluate best model on {args.eval_name}, F1-score: {score:.2f}")


if __name__ == '__main__':
    main()