import argparse
import os
import shutil

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from utils import set_seed, get_collate_fn
from utils import dump2File, saveModelStateDict, evaluate
from prepro import DatasetProcessor
from model import REModel
from torch.cuda.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter


def train(args, model, train_features, benchmarks, save_path, logger, tokenizer):
    train_dataloader = DataLoader(train_features, batch_size=args.train_batch_size, shuffle=True,
                                  collate_fn=get_collate_fn(args.mode, tokenizer), drop_last=True)
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
    for epoch in range(int(args.num_train_epochs)):
        model.zero_grad()
        for step, batch in enumerate(tqdm(train_dataloader)):
            model.train()
            inputs = {'input_ids': batch[0].to(args.device),
                      'attention_mask': batch[1].to(args.device),
                      'labels': batch[2].to(args.device),
                      'ss': batch[3].to(args.device),
                      'os': batch[4].to(args.device),
                      'power': 1 - num_steps / total_steps,
                      }
            loss = model.compute_loss(**inputs) / args.gradient_accumulation_steps
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
                logger.add_scalar("Train Loss", loss.item(), num_steps)

        for tag, features in benchmarks:
            f1 = evaluate(model, features, args.test_batch_size, args.device)
            logger.add_scalar(f"F1-score/{tag}", f1, epoch)
            print(f"epoch: {epoch}, F1-score/{tag}: {f1:.2f}")
            # keep save best model on dev dataset
            if tag == "dev" and f1 > best_score:
                best_score = f1
                print(f"save best model, f1-score: {best_score:.2f}")
                saveModelStateDict(model, os.path.join(save_path, "best.pth"))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", default="./data", type=str)
    parser.add_argument("--split", default="origin", type=str)
    parser.add_argument("--dataset", required=True, type=str)
    parser.add_argument("--model_name", default="roberta-large", type=str)
    parser.add_argument("--input_format", default="typed_entity_marker_punct", type=str,
                        help="in [entity_mask, entity_marker, entity_marker_punct, typed_entity_marker, typed_entity_marker_punct]")

    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
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
    parser.add_argument("--seed", type=int, default=0,
                        help="random seed for initialization")

    parser.add_argument("--dropout_prob", type=float, default=0.1)
    parser.add_argument("--project_name", type=str, default="RE_baseline")
    parser.add_argument("--log_dir", type=str, default="./logs")

    parser.add_argument("--mode", type=str, default="default")
    parser.add_argument("--train_name", type=str, default="train")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    if args.seed >= 0:
        set_seed(args)

    save_path = os.path.join(args.ckpt_dir, args.dataset, args.input_format, args.split,
                             f"{args.model_name}-{args.mode}-{args.train_name}-{args.seed}")
    log_dir = os.path.join(args.log_dir, args.project_name, args.dataset, args.input_format, args.split,
                           f"{args.model_name}-{args.mode}-{args.train_name}-{args.seed}")

    if os.path.exists(log_dir): shutil.rmtree(log_dir)  # 删除历史日志
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    writer = SummaryWriter(log_dir)

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name,
    )

    data_dir = os.path.join(args.data_root, args.dataset, args.split)
    train_file = os.path.join(data_dir, f"{args.train_name}.json")
    dev_file = os.path.join(data_dir, "dev.json")
    test_file = os.path.join(data_dir, "test.json")
    test_challenge_file = os.path.join(data_dir, 'test_challenge.json')

    processor = DatasetProcessor(args, tokenizer)
    train_features = processor.read(train_file)
    dev_features = processor.read(dev_file)
    test_features = processor.read(test_file)
    test_challenge_features = processor.read(test_challenge_file)

    args.num_class = processor.get_num_class()
    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name,
        num_labels=args.num_class,
    )
    config.gradient_checkpointing = True
    config.num_tokens = len(tokenizer)

    # 保存args / config / tokenizer
    dump2File(args, os.path.join(save_path, "args.pkl"))
    dump2File(config, os.path.join(save_path, "config.pkl"))
    # dump2File(tokenizer, os.path.join(save_path, "tokenizer.pkl"))
    dump2File(processor, os.path.join(save_path, "processor.pkl"))

    model = REModel(args, config)
    model.to(args.device)

    benchmarks = (
        ("dev", dev_features),
        ("test", test_features),
        ("test_challenge", test_challenge_features),
    )

    train(args, model, train_features, benchmarks, save_path, writer, tokenizer)


if __name__ == "__main__":
    main()
