"""DPO Trainer — Stage 3: Direct Preference Optimization on RAFT checkpoint.

Uses standard PEFT + transformers (not Unsloth fast forward) because
Unsloth's attention mask patching is incompatible with DPO's 2-pass
forward (causes tensor shape mismatch in attention mask handling).
"""

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, PeftModel
from trl import DPOTrainer, DPOConfig
import torch
import yaml


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def create_model_and_tokenizer(config: dict):
    """Load RAFT checkpoint with standard QLoRA (no Unsloth patching)."""
    model_cfg = config["model"]

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # Load the RAFT adapter — PEFT auto-resolves the base model
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["base_model"],
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["base_model"])

    # Add new LoRA adapter on top for DPO
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()

    return model, tokenizer


def run_dpo(config_path: str):
    """Full DPO training run."""
    from app.training.dpo_dataset import prepare_dpo_dataset

    config = load_config(config_path)

    # Setup W&B if enabled
    if config.get("wandb", {}).get("enabled"):
        import wandb
        wandb.init(
            project=config["wandb"]["project"],
            name=config["wandb"].get("run_name"),
        )

    print("Loading model and tokenizer from RAFT checkpoint...")
    model, tokenizer = create_model_and_tokenizer(config)

    # DPO requires a pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    print("Loading DPO datasets...")
    data_cfg = config["data"]
    train_ds, val_ds = prepare_dpo_dataset(data_cfg["train_file"], data_cfg["val_file"])
    print(f"Train: {len(train_ds)} pairs, Val: {len(val_ds)} pairs")

    train_cfg = config["training"]
    eval_cfg = config["eval"]
    output_cfg = config["output"]
    dpo_cfg = config["dpo"]

    dpo_config = DPOConfig(
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
        beta=dpo_cfg["beta"],
        loss_type=dpo_cfg["loss_type"],
        max_length=config["model"]["max_seq_length"],
        max_prompt_length=256,
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
        gradient_checkpointing=True,
    )

    print("Starting DPO training...")
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        args=dpo_config,
    )

    trainer.train()

    # Save final model
    output_dir = config["output"]["output_dir"]
    print(f"Saving DPO adapter to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save merged model for AWQ export
    print(f"Saving merged model to {output_dir}_merged...")
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(f"{output_dir}_merged")
    tokenizer.save_pretrained(f"{output_dir}_merged")

    print("DPO training complete!")
    return trainer
