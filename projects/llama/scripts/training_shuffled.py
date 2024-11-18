import os
from argparse import ArgumentParser

from datasets import load_from_disk
from dotenv import load_dotenv
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer


def main(args):
    model_id = args.model
    dataset_path = args.dataset
    hg_dataset_test = args.testdataset
    save_model_path = args.output

    os.environ["WANDB_DISABLED"] = "true"
    if args.project is not None:
        os.environ["WANDB_PROJECT"] = args.project

    # loading variables from .env file
    load_dotenv()
    access_token = os.getenv("access_token")

    # TODO: check the lora config parameters, what do they mean and what they impact
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=64,
        lora_alpha=32,
        lora_dropout=0.1,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id, token=access_token, device_map="auto", load_in_4bit=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token

    hg_dataset = load_from_disk(dataset_path)

    def formatting_prompts_func(example):
        output_texts = []
        for i in range(len(example["input"])):
            if isinstance(example["keys"][i], list):
                # noqa: E501
                text = f"### Your goal is to extract structured information from the user's input that matches json format. When extracting information please make sure it matches the type information exactly. Extract the following entities: {example['keys'][i]} Do not add any extra entities.\n\n### Input:\n{example['input'][i]}\n\n ### Output: \n{example['output'][i]}"
                output_texts.append(text)
                continue
            keys = example["keys"][i]
            # noqa: E501
            text = f"### Your goal is to extract structured information from the user's input that matches json format. When extracting information please make sure it matches the type information exactly. Extract the following entities: {keys} Do not add any extra entities.\n\n### Input:\n{example['input'][i]}\n\n ### Output: \n{example['output'][i]}"
            # TODO: Remove "strange" character (guidance returning error)
            text = text.replace(" ", " ")
            text = text.replace("℃", "°C")

            output_texts.append(text)
        return output_texts

    response_template = "### Output:"
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    # TODO: play around with the traning arguments, specifically the optimizer ones (learning rate, batch size, epochs)
    args = TrainingArguments(
        report_to="wandb",
        output_dir=save_model_path,
        learning_rate=2e-4,
        num_train_epochs=1,
        logging_steps=10,
        gradient_checkpointing=True,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=2,
        do_eval=True,
        eval_steps=10,
        evaluation_strategy="steps",
    )

    # TODO: check the SFTTrainer parameters and how to train the model
    # Default optimizer is AdamW because its default in transformers.Trainer.
    # SFTTrainer is a wraper around transformers.Trainer and inherits all of its attributes and methods.
    # To update weights every n batch iterations use gradient_accumulation_steps!
    trainer = SFTTrainer(
        model,
        train_dataset=hg_dataset,
        eval_dataset=hg_dataset_test,
        formatting_func=formatting_prompts_func,
        data_collator=collator,
        max_seq_length=4096,
        args=args,
        peft_config=peft_config,
    )

    trainer.train()

    trainer.save_model(save_model_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--model", type=str, default=None, help="model_id (hugging face)"
    )
    parser.add_argument(
        "--train_data", type=str, default=None, help="path to training dataset"
    )
    parser.add_argument(
        "--test_data", type=str, default=None, help="path to test dataset"
    )
    parser.add_argument(
        "--output", type=str, default=None, help="path for saving the model"
    )
    parser.add_argument("--project", type=str, default=None, help="wandb project")
    args = parser.parse_args()

    main(args)
