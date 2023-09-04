from argparse import ArgumentParser
import os
from utils import load4File, loadModel, evaluate
from model import REModel


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

    args = parser.parse_args()
    save_path = os.path.join(args.ckpt_dir, args.dataset, args.input_format,f"{args.model_name}-{args.seed}")
    __args = load4File(os.path.join(save_path, "args.pkl"))
    config = load4File(os.path.join(save_path, "config.pkl"))
    processor = load4File(os.path.join(save_path, "processor.pkl"))

    # load model
    model = REModel(__args, config)
    model.to(0)
    model.load_state_dict(loadModel(os.path.join(save_path, "best.pth")))

    eval_file = os.path.join(args.data_root, args.dataset, args.eval_name + ".json")
    eval_features = processor.read(eval_file)

    score = evaluate(__args, model, eval_features)
    print(f"evaluate best model on {args.eval_name}, F1-score: {score:.2f}")


if __name__ == '__main__':
    main()