import argparse
import pandas as pd
from datasets import Dataset
from huggingface_hub import login
from peft import (
    get_peft_model,
    prepare_model_for_kbit_training,
    LoraConfig,
    TaskType,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

def tokenize_bbq(element):
    outputs = tokenizer(
        [i+ " " + j for i,j in zip(element["prompt"], element["chosen"])],
        truncation=True,
        max_length=context_length,
        padding="max_length",
        return_overflowing_tokens=True,
        return_length=True,
    )
    input_batch = []
    for input_ids in outputs["input_ids"]:
        input_batch.append(input_ids)
    return {"input_ids": input_batch}

def tokenize_sft(element):
    text = [
        element["prompt"][i] + " " + element["chosen"][i]
        for i in range(len(element["prompt"]))
    ]
    outputs = tokenizer(
        text,
        truncation=True,
        max_length=context_length,
        padding="max_length",
        return_overflowing_tokens=True,
        return_length=True,
    )
    input_batch = []
    for input_ids in outputs["input_ids"]:
        input_batch.append(input_ids)
    return {"input_ids": input_batch}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="Model",
    )
    parser.add_argument(
        "-token",
        type=str,
        default="",
        help="Huggingface token that grants access to Llama model",
    )
    parser.add_argument(
        "-seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "-dataset",
        type=str,
        default="bbqsft",
        help="Dataset to fine-tune on",
    )
    parser.add_argument(
        '--cache_path', default='')
    args = parser.parse_args()
    login(args.token)
    # load configs for QLora
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules="all-linear",
        bias="none",
        r=64,
        lora_alpha=16,
        lora_dropout=0.1,
    )
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="float16",
    )
    # load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=args.cache_path)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    if "gemma" in args.model:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            quantization_config=bnb_config,
            device_map="auto",
            attn_implementation="eager",
            trust_remote_code=True,
            cache_dir=args.cache_path
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            cache_dir=args.cache_path
        )
    # prepare model
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    model.enable_input_require_grads()
    print("MODEL LOADED")
    ################################################################################
    #    LOAD DATA
    ################################################################################
    # SFT training
    dataset = pd.read_json("[folder_path]/sft_train.jsonl", lines=True)
    num_epochs = 1
    n_steps = 100
    dataset = Dataset.from_pandas(dataset).train_test_split(
        test_size=0.05, seed=args.seed
    )

    context_length = 512
    tokenized_datasets = dataset.map(
        tokenize_sft,
        batched=True,
        remove_columns=dataset["train"].column_names,
    )
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    print("DATA LOADED")
    training_args = TrainingArguments(
        output_dir=f"{args.model}_lora_{args.dataset}",
        num_train_epochs=num_epochs,
        save_total_limit=5,
        eval_strategy="steps",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        warmup_steps=n_steps,
        weight_decay=0.001,
        metric_for_best_model="eval_loss",
        fp16=True,
        remove_unused_columns=False,
        logging_steps=500,
        eval_steps=500,
        save_steps=500,
        save_strategy="steps",
        learning_rate=3e-4,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": True},
        load_best_model_at_end=True,
        seed=args.seed,
    )
    model = get_peft_model(model, peft_config)
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        data_collator=data_collator,
    )
    print("START TRAINING")
    trainer.train()
    model = trainer.model.merge_and_unload()
    model.save_pretrained(f"[cache folder]/{args.model}")
    tokenizer.save_pretrained(f"[cache folder]/{args.model}")