"""
Finetune CodeT5+ models on any Seq2Seq LM tasks
You can customize your own training data by following the HF dataset format to cache it to args.cache_data
Author: Yue Wang
Date: June 2023
"""

import os
import pprint
import argparse
# from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, TrainingArguments, Trainer, set_seed

seed = 42
set_seed(seed)


def run_training(args, model, train_data, eval_data):
    print(f"Starting main loop")

    training_args = TrainingArguments(
        report_to='none',
        output_dir=args.save_dir,
        overwrite_output_dir=True,

        do_train=True,
        # do_eval=True,
        # evaluation_strategy='epoch',
        save_strategy='epoch',

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
        fp16=args.fp16,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
    )

    trainer.train()

    if args.local_rank in [0, -1]:
        final_checkpoint_dir = os.path.join(args.save_dir, "final_checkpoint")
        model.save_pretrained(final_checkpoint_dir)
        print(f'  ==> Finish training and save to {final_checkpoint_dir}')


def load_tokenize_data(args, tokenizer):
    train_dataset = 1  # load_dataset("csv", data_files="./datasets/train.csv", split="train")
    eval_dataset = 1  # load_dataset("csv", data_files="./datasets/validation.csv", split="train")

    def preprocess_function(examples):
        source = examples["src"]
        target = [str(t) for t in examples["tgt"]]

        model_inputs = tokenizer(source, max_length=args.max_source_len, padding="max_length", truncation=True)
        labels = tokenizer(target, max_length=args.max_target_len, padding="max_length", truncation=True)

        model_inputs["labels"] = labels["input_ids"].copy()
        model_inputs["labels"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in model_inputs["labels"]
        ]
        return model_inputs

    train_data = train_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=train_dataset.column_names,
        num_proc=64,
        load_from_cache_file=False,
    )
    print(f'  ==> Loaded {len(train_data)} samples for training')

    eval_data = eval_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=eval_dataset.column_names,
        num_proc=64,
        load_from_cache_file=False,
    )
    print(f'  ==> Loaded {len(eval_data)} samples for validation')

    return train_data, eval_data


def main(args):
    argsdict = vars(args)
    print(pprint.pformat(argsdict))

    # Save command to file
    with open(os.path.join(args.save_dir, "command.txt"), 'w') as f:
        f.write(pprint.pformat(argsdict))

    # Load and tokenize data using the tokenizer from `args.load`. If the data is already cached, load it from there.
    # You can customize this function to load your own data for any Seq2Seq LM tasks.
    tokenizer = AutoTokenizer.from_pretrained(args.load)
    train_data, eval_data = load_tokenize_data(args, tokenizer)
    train_data.shuffle(seed=seed)
    eval_data.shuffle(seed=seed)

    if args.data_num != -1:
        train_data = train_data.select([i for i in range(args.data_num)])

    # Load model from `args.load`
    model = AutoModelForSeq2SeqLM.from_pretrained(args.load)
    print(f"  ==> Loaded model from {args.load}, model size {model.num_parameters()}")

    # model.config.decoder_start_token_id = tokenizer.bos_token_id
    # model.config.pad_token_id = tokenizer.pad_token_id

    run_training(args, model, train_data, eval_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CodeT5+ finetuning on Seq2Seq LM task")
    parser.add_argument('--data-num', default=-1, type=int)
    parser.add_argument('--max-source-len', default=512, type=int)
    parser.add_argument('--max-target-len', default=4, type=int)
    parser.add_argument('--load', default='Salesforce/codet5p-220m', type=str)

    # Training
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--lr', default=5e-5, type=float)
    parser.add_argument('--batch-size-per-replica', default=32, type=int)
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--fp16', default=True, action='store_true')

    # Logging and stuff
    parser.add_argument('--save-dir', default="saved_models", type=str)
    parser.add_argument('--log-freq', default=10, type=int)
    parser.add_argument('--save-freq', default=500, type=int)

    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    main(args)
