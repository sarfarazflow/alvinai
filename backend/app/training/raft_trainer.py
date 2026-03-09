"""RAFT Trainer — Stage 2: Retrieval-Augmented Fine-Tuning with QLoRA on SFT checkpoint."""

from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
import yaml


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def create_model_and_tokenizer(config: dict):
    """Load SFT checkpoint with QLoRA via Unsloth."""
    model_cfg = config["model"]
    lora_cfg = config["lora"]

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_cfg["base_model"],
        max_seq_length=model_cfg["max_seq_length"],
        load_in_4bit=model_cfg["load_in_4bit"],
        dtype=None,  # auto-detect
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"],
        bias="none",
        use_gradient_checkpointing="unsloth",
    )

    return model, tokenizer


def create_training_args(config: dict) -> TrainingArguments:
    """Build TrainingArguments from config."""
    train_cfg = config["training"]
    eval_cfg = config["eval"]
    output_cfg = config["output"]

    return TrainingArguments(
        output_dir=output_cfg["output_dir"],
        num_train_epochs=train_cfg["num_epochs"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        warmup_ratio=train_cfg["warmup_ratio"],
        weight_decay=train_cfg["weight_decay"],
        max_grad_norm=train_cfg["max_grad_norm"],
        bf16=train_cfg.get("bf16", False),
        fp16=train_cfg.get("fp16", False),
        seed=train_cfg["seed"],
        eval_strategy=eval_cfg["eval_strategy"],
        eval_steps=eval_cfg["eval_steps"],
        save_strategy=eval_cfg["save_strategy"],
        save_steps=eval_cfg["save_steps"],
        save_total_limit=eval_cfg["save_total_limit"],
        load_best_model_at_end=eval_cfg["load_best_model_at_end"],
        metric_for_best_model=eval_cfg["metric_for_best_model"],
        logging_dir=output_cfg["logging_dir"],
        logging_steps=output_cfg["logging_steps"],
        report_to="wandb" if config.get("wandb", {}).get("enabled") else "none",
    )


def run_raft(config_path: str):
    """Full RAFT training run."""
    from app.training.raft_dataset import prepare_raft_dataset

    config = load_config(config_path)

    # Setup W&B if enabled
    if config.get("wandb", {}).get("enabled"):
        import wandb
        wandb.init(
            project=config["wandb"]["project"],
            name=config["wandb"].get("run_name"),
        )

    print("Loading model and tokenizer from SFT checkpoint...")
    model, tokenizer = create_model_and_tokenizer(config)

    print("Loading RAFT datasets...")
    data_cfg = config["data"]
    train_ds, val_ds = prepare_raft_dataset(data_cfg["train_file"], data_cfg["val_file"])
    print(f"Train: {len(train_ds)} examples, Val: {len(val_ds)} examples")

    print("Starting RAFT training...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        dataset_text_field="text",
        max_seq_length=config["model"]["max_seq_length"],
        args=create_training_args(config),
    )

    trainer.train()

    # Save final model
    output_dir = config["output"]["output_dir"]
    print(f"Saving RAFT adapter to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save merged model for next stage (DPO)
    print(f"Saving merged model to {output_dir}_merged...")
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(f"{output_dir}_merged")
    tokenizer.save_pretrained(f"{output_dir}_merged")

    print("RAFT training complete!")
    return trainer
