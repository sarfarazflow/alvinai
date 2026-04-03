"""DPO Trainer — Stage 3: Direct Preference Optimization on RAFT checkpoint.

Uses Unsloth for model loading with instance-level forward patch to
handle DPO's 4D attention masks.
"""

# Stub mergekit/llm_blender: TRL optionally imports these but we don't need them.
# Auto-creates any submodule on access so we don't play whack-a-mole.
import sys
import types
import importlib.machinery

class _StubModule(types.ModuleType):
    """Module stub that auto-creates submodules on attribute access."""
    def __getattr__(self, name):
        fullname = f"{self.__name__}.{name}"
        if fullname not in sys.modules:
            sub = _StubModule(fullname)
            sub.__file__ = f"<stub {fullname}>"
            sub.__path__ = []
            sub.__package__ = self.__name__
            sub.__spec__ = importlib.machinery.ModuleSpec(fullname, None, origin=sub.__file__)
            sys.modules[fullname] = sub
        return sys.modules[fullname]

for _mod_name in ("mergekit", "llm_blender"):
    if _mod_name not in sys.modules:
        _mod = _StubModule(_mod_name)
        _mod.__file__ = f"<stub {_mod_name}>"
        _mod.__path__ = []
        _mod.__package__ = _mod_name
        _mod.__spec__ = importlib.machinery.ModuleSpec(_mod_name, None, origin=_mod.__file__)
        sys.modules[_mod_name] = _mod

from unsloth import FastLanguageModel, PatchDPOTrainer
from trl import DPOTrainer, DPOConfig
import torch
import yaml

PatchDPOTrainer()


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _patch_model_for_dpo(model):
    """Patch the model's inner LlamaModel to handle 4D attention masks from DPO.

    Unsloth's LlamaModel_fast_forward expects 2D attention_mask [batch, seq]
    but DPO's TRL creates 4D causal masks [batch, 1, seq, seq]. We intercept
    the inner model's forward to convert 4D→2D before Unsloth processes it.
    """
    # Navigate to the inner model (PeftModel → base_model → model → model)
    inner_model = model
    for attr in ["model", "model", "model"]:
        if hasattr(inner_model, attr):
            inner_model = getattr(inner_model, attr)

    original_forward = inner_model.forward

    def patched_forward(*args, **kwargs):
        attention_mask = kwargs.get("attention_mask", None)
        if attention_mask is not None and attention_mask.dim() > 2:
            # Convert ND mask to 2D: take the last row of each head's mask
            # This extracts the padding pattern from the causal mask
            if attention_mask.dim() == 4:
                # [batch, heads, seq_q, seq_k] → [batch, seq_k]
                kwargs["attention_mask"] = attention_mask[:, 0, -1, :]
            elif attention_mask.dim() == 3:
                # [batch, seq_q, seq_k] → [batch, seq_k]
                kwargs["attention_mask"] = attention_mask[:, -1, :]
            # Convert from float (0/-inf) to int (1/0) if needed
            mask = kwargs["attention_mask"]
            if mask.dtype in (torch.float16, torch.bfloat16, torch.float32):
                kwargs["attention_mask"] = (mask != float("-inf")).to(torch.long)
                # If mask has very negative values instead of -inf
                if kwargs["attention_mask"].sum() == 0:
                    kwargs["attention_mask"] = (mask > -1.0).to(torch.long)
        return original_forward(*args, **kwargs)

    inner_model.forward = patched_forward
    return model


def create_model_and_tokenizer(config: dict):
    """Load RAFT checkpoint via Unsloth."""
    model_cfg = config["model"]

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_cfg["base_model"],
        max_seq_length=model_cfg["max_seq_length"],
        load_in_4bit=model_cfg["load_in_4bit"],
        dtype=None,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
        use_gradient_checkpointing="unsloth",
    )

    # Patch the model to handle DPO's 4D attention masks
    model = _patch_model_for_dpo(model)

    return model, tokenizer


def run_dpo(config_path: str):
    """Full DPO training run."""
    from app.training.dpo_dataset import prepare_dpo_dataset

    config = load_config(config_path)

    if config.get("wandb", {}).get("enabled"):
        import wandb
        wandb_cfg = config["wandb"]
        wandb.init(
            project=wandb_cfg["project"],
            name=wandb_cfg.get("run_name"),
            group=wandb_cfg.get("group"),
            tags=wandb_cfg.get("tags"),
        )

    print("Loading model and tokenizer from RAFT checkpoint...")
    model, tokenizer = create_model_and_tokenizer(config)

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

    output_dir = config["output"]["output_dir"]
    print(f"Saving DPO adapter to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"Saving merged model to {output_dir}_merged...")
    model.save_pretrained_merged(f"{output_dir}_merged", tokenizer, save_method="merged_16bit")

    print("DPO training complete!")
    return trainer
