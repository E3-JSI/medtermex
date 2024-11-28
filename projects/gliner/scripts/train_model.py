import json
import random
from argparse import ArgumentParser
from pathlib import Path

import torch
from gliner import GLiNER
from gliner.data_processing.collator import DataCollator
from gliner.training import Trainer, TrainingArguments


def main(args):
    if not Path(args.data_train_file).exists():
        raise FileNotFoundError(
            f"Train data file {args.data_train_file} does not exist"
        )

    # load the training data file
    with open(args.data_train_file, "r", encoding="utf8") as f:
        data = json.load(f)

    # prepare the data examples
    random.shuffle(data)
    train_data_size = int(len(data) * args.train_data_ratio)
    train_dataset = data[:train_data_size]
    eval_dataset = data[train_data_size:]

    # load the device (GPU or CPU)
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.use_cpu else "cpu"
    )

    # load the model
    model = GLiNER.from_pretrained(args.model_name_or_path)

    # prepare the data collator
    data_collator = DataCollator(
        model.config, data_processor=model.data_processor, prepare_labels=True
    )

    # train the model
    model.to(device)
    training_args = TrainingArguments(
        output_dir=args.training_output_dir,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        others_lr=args.other_lr,
        others_weight_decay=args.other_weight_decay,
        lr_scheduler_type="linear",
        warmup_ratio=0.1,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        focal_loss_alpha=0.75,
        focal_loss_gamma=2,
        num_train_epochs=args.num_train_epochs,
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=model.data_processor.transformer_tokenizer,
        data_collator=data_collator,
    )
    trainer.train()

    # save the model
    model.save_pretrained(args.model_output_dir)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--data_train_file",
        type=str,
        help="path to the training data file",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="path to the pre-trained model or model name",
    )
    parser.add_argument(
        "--model_output_dir",
        type=str,
        help="path to the output directory for the trained model",
    )
    parser.add_argument(
        "--training_output_dir",
        type=str,
        default="models/tmp",
        help="path to the output directory for the trained model",
    )
    parser.add_argument(
        "--train_data_ratio",
        type=float,
        default=0.8,
        help="ratio of the training data",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=3,
        help="number of training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="batch size",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,
        help="learning rate",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="weight decay",
    )
    parser.add_argument(
        "--other_lr",
        type=float,
        default=1e-5,
        help="learning rate for other parameters",
    )
    parser.add_argument(
        "--other_weight_decay",
        type=float,
        default=0.01,
        help="weight decay for other parameters",
    )
    parser.add_argument(
        "--use_cpu",
        action="store_true",
        help="whether to use CPU (default is GPU)",
    )

    args = parser.parse_args()
    main(args)
