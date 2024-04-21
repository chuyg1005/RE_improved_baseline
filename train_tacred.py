import argparse
import os, json
import shutil

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from utils import set_seed, get_collate_fn
from utils import dump2File, saveModelStateDict, evaluate, load4File, buildModel
from prepro import DatasetProcessor, get_label_to_id
from torch.cuda.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter


def train(args, model, train_features, benchmarks, save_path, logger):
    train_dataloader = DataLoader(train_features, batch_size=args.train_batch_size, shuffle=True,
                                  collate_fn=get_collate_fn(args.mode), drop_last=True)
    total_steps = int(len(train_dataloader) * args.num_train_epochs // args.gradient_accumulation_steps)
    warmup_steps = int(total_steps * args.warmup_ratio)

    scaler = GradScaler()
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                num_training_steps=total_steps)
    print('Total steps: {}'.format(total_steps))
    print('Warmup steps: {}'.format(warmup_steps))

    num_steps = 0
    best_score = 0
    log_step = 0
    for epoch in range(int(args.num_train_epochs)):
        model.zero_grad()
        for step, batch in enumerate(tqdm(train_dataloader)):
            model.train()
            inputs = {'input_ids': batch[0].to(args.device),
                      'attention_mask': batch[1].to(args.device),
                      'labels': batch[2].to(args.device),
                      'ss': batch[3].to(args.device),
                      'os': batch[4].to(args.device),
                      # 'power': 1 - num_steps / total_steps * (1 - args.alpha),
                      }
            loss = model.compute_loss(**inputs)
            log_step += 1
            loss = loss / args.gradient_accumulation_steps
            scaler.scale(loss).backward()
            if step % args.gradient_accumulation_steps == 0:
                num_steps += 1
                if args.max_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                model.zero_grad()
                if logger: logger.add_scalar("Train Loss", loss.item(), num_steps)

                # evaluate
                if num_steps % args.eval_steps == 0:
                    for tag, features in benchmarks:
                        f1 = evaluate(model, features, args.test_batch_size, args.device)
                        if logger: logger.add_scalar(f"F1-score/{tag}", f1, num_steps)
                        print(f"epoch: {epoch}, F1-score/{tag}: {f1:.2f}")
                        # keep save best model on dev dataset
                        if tag[:3] == "dev" and f1 > best_score:
                            best_score = f1
                            print(f"save best model, f1-score: {best_score:.2f}")
                            if save_path: saveModelStateDict(model, os.path.join(save_path, "best.pth"))

        # break  # only train one epoch

    for tag, features in benchmarks:
        if features is None: continue
        f1 = evaluate(model, features, args.test_batch_size, args.device)
        # keep save best model on dev dataset
        if tag[:3] == "dev" and f1 > best_score:
            best_score = f1
            print(f"save best model, f1-score: {best_score:.2f}")
            if save_path: saveModelStateDict(model, os.path.join(save_path, "best.pth"))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", default="./data", type=str)
    parser.add_argument("--dataset", required=True, type=str)
    parser.add_argument("--model_name", default="roberta-large", type=str)
    parser.add_argument("--input_format", default="typed_entity_marker_punct", type=str,
                        help="in [entity_mask, entity_marker, entity_marker_punct, typed_entity_marker, typed_entity_marker_punct]")

    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated.")

    parser.add_argument("--ckpt_dir", default="./ckpts", type=str, help="model checkpoints save directories.")

    parser.add_argument("--train_batch_size", default=64, type=int,
                        help="Batch size for training.")
    parser.add_argument("--test_batch_size", default=128, type=int,
                        help="Batch size for testing.")
    parser.add_argument("--learning_rate", default=3e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int,
                        help="Number of updates steps to accumulate the gradients for, before performing a "
                             "backward/update pass.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--warmup_ratio", default=0.1, type=float,
                        help="Warm up ratio for Adam.")
    parser.add_argument("--num_train_epochs", default=5.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument("--dropout_prob", type=float, default=0.1)
    parser.add_argument("--project_name", type=str, default="RE_baseline")
    parser.add_argument("--log_dir", type=str, default="./logs")

    parser.add_argument("--mode", type=str, default="default")
    parser.add_argument("--train_name", type=str, default="train4debias")

    parser.add_argument("--alpha", type=float, default=0., help="")

    parser.add_argument("--eval_steps", type=int, default=1000, help="")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    set_seed(args)

    save_path = os.path.join(args.ckpt_dir, args.dataset, f'{args.mode}-{args.seed}')
    log_dir = os.path.join(args.log_dir, args.project_name, args.dataset, f'{args.mode}-{args.seed}')
    cache_dir = os.path.join(args.ckpt_dir, args.dataset, "cache")  # 缓存中间结果
    os.makedirs(cache_dir, exist_ok=True)

    # save_path = log_dir = None

    if log_dir and os.path.exists(log_dir): shutil.rmtree(log_dir)  # 删除历史日志
    if save_path: os.makedirs(save_path, exist_ok=True)
    if log_dir: os.makedirs(log_dir, exist_ok=True)

    writer = SummaryWriter(log_dir) if log_dir else None

    data_dir = os.path.join(args.data_root, args.dataset)
    train_file = os.path.join(data_dir, f"{args.train_name}.json")

    if args.mode == 'EntityOnly':  # entity only
        files = {
            "train": train_file,
            "dev-eo": os.path.join(data_dir, "dev-eo.json"),
            "test-eo": os.path.join(data_dir, "test-eo.json"),
        }
    elif args.mode == 'EntityMask':  # context only
        files = {
            "train": train_file,
            "dev-co": os.path.join(data_dir, "dev-co.json"),
            "test-co": os.path.join(data_dir, "test-co.json"),
        }
    else:
        files = {
            "train": train_file,
            "dev": os.path.join(data_dir, "dev.json"),
            "test": os.path.join(data_dir, "test.json"),
        }

    label2id = get_label_to_id(args.dataset)
    args.num_class = len(label2id)

    # 保存args / config / tokenizer
    if save_path is not None:
        dump2File(vars(args), os.path.join(save_path, "args.json"))

    model, tokenizer = buildModel(args)
    model.to(args.device)

    processor = DatasetProcessor(args, tokenizer)

    features = {}
    for tag in files.keys():
        filepath = files[tag]
        if os.path.exists(os.path.join(cache_dir, f"{tag}_features.json")):
            print(f"loading {tag} features from cache")
            features[tag] = load4File(os.path.join(cache_dir, f"{tag}_features.json"))
        else:
            print(f"processing {tag} features")
            features[tag] = processor.read(filepath)
            dump2File(features[tag], os.path.join(cache_dir, f"{tag}_features.json"))

    train_features = features["train"]
    benchmarks = [(tag, features) for tag, features in features.items() if tag != "train"]

    train(args, model, train_features, benchmarks, save_path, writer)


if __name__ == "__main__":
    main()
