import argparse
import pprint
import os
from typing import Tuple

from datasets import Dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer

import get_oracles_dataset
import get_tokens_dataset


def load_dataset(tokenizer, dataset_name: str) -> Tuple[Dataset, Dataset]:
    if dataset_name == "oracles":
        train_dataset = get_oracles_dataset.get_custom_dataset(None, tokenizer, "train")
        validation_dataset = get_oracles_dataset.get_custom_dataset(None, tokenizer, "validation")
    elif dataset_name == "tokens":
        train_dataset = get_tokens_dataset.get_custom_dataset(None, tokenizer, "train")
        validation_dataset = get_tokens_dataset.get_custom_dataset(None, tokenizer, "validation")
    else:
        raise ValueError("Unrecognized dataset name", dataset_name)
    return train_dataset, validation_dataset


def run_train(args, model, train_data, eval_data):
    print(f"Starting main loop")
    train_args = TrainingArguments(
        report_to=["none"],
        output_dir=args.save_dir,
        overwrite_output_dir=True,
        do_train=True,
        # do_eval=True,
        # evaluation_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size_per_replica,
        per_device_eval_batch_size=args.batch_size_per_replica * 4,
        learning_rate=args.lr,
        logging_dir=args.save_dir,
        logging_first_step=True,
        logging_steps=args.log_freq,
        # save_total_limit=1,
        # load_best_model_at_end=True,
        dataloader_drop_last=True,
        dataloader_num_workers=4,
        local_rank=args.local_rank,
        fp16=args.fp16
    )
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_data,
        eval_dataset=eval_data
    )
    trainer.train()
    if args.local_rank in [0, -1]:
        final_checkpoint_dir = os.path.join(args.save_dir, "final_checkpoint")
        model.save_pretrained(final_checkpoint_dir)
        print(f"  ==> Finish training and save to {final_checkpoint_dir}")


def main(args):
    argsdict = vars(args)
    print(pprint.pformat(argsdict))

    # setup dataset and models
    tokenizer = AutoTokenizer.from_pretrained(args.load)
    train_dataset, eval_dataset = load_dataset(tokenizer, args.dataset_name)
    train_dataset.shuffle()
    eval_dataset.shuffle()
    model = AutoModelForCausalLM.from_pretrained(
        args.load,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    print(f"  ==> Loaded model from {args.load}, model size {model.num_parameters()}")

    model.config.decoder_start_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    run_train(args, model, train_dataset, eval_dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CodeLLaMa fine-tuning on test oracles")
    parser.add_argument("--dataset-name", default="oracles", type=str)
    parser.add_argument("--max-source-len", default=4096, type=int)
    parser.add_argument("--max-target-len", default=4, type=int)
    parser.add_argument("--load", default="codellama/CodeLlama-7b-hf", type=str)

    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--lr", default=5e-5, type=float)
    parser.add_argument("--batch-size-per-replica", default=32, type=int)
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--fp16", default=True, action="store_true")

    parser.add_argument("--save-dir", default="saved_models", type=str)
    parser.add_argument("--log-freq", default=10, type=int)
    parser.add_argument("--save-freq", default=500, type=int)

    argv = parser.parse_args()

    os.makedirs(argv.save_dir, exist_ok=True)

    main(argv)
