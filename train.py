import wandb
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    Trainer,
    TrainingArguments
)
from datasets import Dataset

# -----------------------
# Tokenizer
# -----------------------
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
# Use default GPT-2 EOS and set PAD to EOS for causal LM training
tokenizer.pad_token = tokenizer.eos_token

# -----------------------
# Build regular prompt→completion dataset (no roles)
# -----------------------
def build_example(prompt, completion):
    # Compose sample without any role prefixes.
    # We keep instruction tuning style by masking the prompt, so the model learns
    # to generate the completion given the prompt.
    text = f"{prompt}\n{completion}{tokenizer.eos_token}"

    tokens = tokenizer(text, truncation=True)
    input_ids = tokens["input_ids"]

    # Mask the prompt part; only learn on the completion and EOS
    labels = [-100] * len(input_ids)
    prompt_prefix = f"{prompt}\n"
    prefix_ids = tokenizer(prompt_prefix, truncation=True)["input_ids"]
    start = len(prefix_ids)
    labels[start:] = input_ids[start:]

    return {
        "input_ids": input_ids,
        "attention_mask": tokens["attention_mask"],
        "labels": labels,
    }

def load_examples_from_file(path):
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    # Split on blank lines to get blocks, no reliance on any role markers
    blocks = [b.strip() for b in content.split("\n\n")]
    examples = []

    for block in blocks:
        if not block:
            continue

        lines = [line.strip() for line in block.splitlines() if line.strip()]

        # Expect exactly two lines per example: first = prompt, second = completion
        if len(lines) < 2:
            continue

        prompt = lines[0]
        completion = lines[1]

        examples.append(build_example(prompt, completion))

    return examples


examples = load_examples_from_file("train.txt")
dataset = Dataset.from_list(examples)

# -----------------------
# Model
# -----------------------
config = GPT2Config(
    vocab_size=len(tokenizer),
    n_positions=512,
    n_ctx=512,
    n_embd=768,
    n_layer=8,
    n_head=12,
)

model = GPT2LMHeadModel(config)
model.resize_token_embeddings(len(tokenizer))


def print_parameter_count(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")


print_parameter_count(model)

# -----------------------
# W&B Logging
# -----------------------
wandb.init(project="axiom-base-llm", name="tinyLLM-100M-run", config={
    "vocab_size": len(tokenizer),
    "n_positions": config.n_positions,
    "n_embd": config.n_embd,
    "n_layer": config.n_layer,
    "n_head": config.n_head,
    "total_params": sum(p.numel() for p in model.parameters()),
})

# -----------------------
# Training
# -----------------------
args = TrainingArguments(
    output_dir="tinyLLM",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,       # Effective batch size = 8; helps stabilize 100M training
    num_train_epochs=50,                 # Reduced from 200; 100M params overfit fast on small data
    learning_rate=1e-4,                  # Lower LR is safer for larger models
    weight_decay=0.01,                   # Regularization to prevent overfitting
    max_grad_norm=1.0,                   # Gradient clipping for training stability
    warmup_steps=200,                    # Slowly ramp LR to prevent early divergence
    lr_scheduler_type="cosine",          # Smooth decay after warmup
    logging_steps=10,
    save_steps=500,
    save_total_limit=2,
    fp16=True,                           # Mixed precision: cuts VRAM ~40% and speeds up training
    report_to="wandb",
    dataloader_num_workers=0,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset,
)

trainer.train()

model.save_pretrained("tinyLLM")
tokenizer.save_pretrained("tinyLLM")
