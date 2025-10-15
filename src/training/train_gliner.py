import json
import random
from argparse import ArgumentParser
from pathlib import Path

import torch
from gliner import GLiNER
from gliner.data_processing.collator import DataCollator
from gliner.training import Trainer, TrainingArguments

from src.core.utils.argument_parsers import str2bool

# ===============================
# Main
# ===============================


def get_args():
    parser = ArgumentParser("Train a GLiNER model")
    parser.add_argument("--train-dataset-file", type=str, help="The path to the training dataset file")
    parser.add_argument("--model-name-or-path", type=str, help="The path to the pre-trained model or the model name")
    parser.add_argument("--model-output-dir", type=str, help="The path to the output directory for the trained model")
    parser.add_argument(
        "--training-output-dir",
        type=str,
        default="models/tmp",
        help="The path to the output directory for the trained model",
    )
    parser.add_argument(
        "--train-validation-ratio", type=float, default=0.8, help="The ratio of the training/validation data"
    )
    parser.add_argument("--train-num-epochs", type=int, default=3, help="The number of training epochs")
    parser.add_argument("--train-batch-size", type=int, default=8, help="The batch size")
    parser.add_argument("--train-learning-rate", type=float, default=5e-6, help="The learning rate")
    parser.add_argument("--train-weight-decay", type=float, default=0.01, help="The weight decay")
    parser.add_argument("--train-other-lr", type=float, default=1e-5, help="The learning rate for other parameters")
    parser.add_argument(
        "--train-other-weight-decay", type=float, default=0.01, help="The weight decay for other parameters"
    )
    parser.add_argument("--use-cpu", type=str2bool, default=False, help="Whether to use the CPU (default is GPU)")

    return parser.parse_args()


def main(args):
    if not Path(args.train_dataset_file).exists():
        raise FileNotFoundError(f"Train data file {args.train_dataset_file} does not exist")

    # load the training data file
    with open(args.train_dataset_file, "r", encoding="utf8") as f:
        data = json.load(f)

    # prepare the data examples
    random.shuffle(data)
    if args.train_validation_ratio > 0:
        train_data_size = int(len(data) * args.train_validation_ratio)
        train_dataset = data[:train_data_size]
        valid_dataset = data[train_data_size:]
    else:
        train_dataset = data
        valid_dataset = None

    # load the device (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.use_cpu else "cpu")

    # load the model
    model = GLiNER.from_pretrained(args.model_name_or_path)

    # prepare the data collator
    data_collator = DataCollator(model.config, data_processor=model.data_processor, prepare_labels=True)

    # train the model
    model.to(device)
    training_args = TrainingArguments(
        output_dir=args.training_output_dir,
        learning_rate=args.train_learning_rate,
        weight_decay=args.train_weight_decay,
        others_lr=args.train_other_lr,
        others_weight_decay=args.train_other_weight_decay,
        lr_scheduler_type="linear",
        warmup_ratio=0.1,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.train_batch_size,
        focal_loss_alpha=0.75,
        focal_loss_gamma=2,
        num_train_epochs=args.train_num_epochs,
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=model.data_processor.transformer_tokenizer,
        data_collator=data_collator,
    )
    trainer.train()

    # save the model
    model.save_pretrained(args.model_output_dir)


if __name__ == "__main__":
    args = get_args()
    main(args)
