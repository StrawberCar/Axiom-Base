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
# Load the whole dataset and chunk into fixed-length blocks
# -----------------------
BLOCK_SIZE = 1024

print("Reading train.txt...", flush=True)
with open("train.txt", "r", encoding="utf-8") as f:
    text = f.read()
print(f"Read {len(text):,} characters", flush=True)

print("Tokenizing...", flush=True)
all_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
print(f"Tokenized into {len(all_ids):,} tokens", flush=True)

# Drop the remainder so every block is exactly BLOCK_SIZE tokens
n_blocks = len(all_ids) // BLOCK_SIZE
trimmed_ids = all_ids[: n_blocks * BLOCK_SIZE]
print(f"Chunking into {n_blocks:,} blocks of {BLOCK_SIZE} tokens...", flush=True)

input_ids = [trimmed_ids[i : i + BLOCK_SIZE] for i in range(0, len(trimmed_ids), BLOCK_SIZE)]

dataset = Dataset.from_dict({
    "input_ids": input_ids,
    "attention_mask": [[1] * BLOCK_SIZE for _ in input_ids],
    "labels": [ids[:] for ids in input_ids],
})

print(f"Dataset ready: {len(dataset)} blocks of {BLOCK_SIZE} tokens ({len(dataset) * BLOCK_SIZE:,} tokens total)", flush=True)

# -----------------------
# Model
# -----------------------
config = GPT2Config(
    vocab_size=len(tokenizer),
    n_positions=1024,
    n_ctx=1024,
    n_embd=1024,
    n_layer=24,
    n_head=16,
)

model = GPT2LMHeadModel(config)
model.resize_token_embeddings(len(tokenizer))

def print_parameter_count(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

print_parameter_count(model)


wandb.init(project="axiom-base-llm", name="tinyLLM-350M-run", config={
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
    overwrite_output_dir=True,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=2,
    num_train_epochs=3,
    learning_rate=3e-4,
    warmup_steps=200,
    logging_steps=10,
    save_steps=500,
    save_total_limit=1,
    fp16=True,
    gradient_checkpointing=False,
    report_to="wandb",
    dataloader_num_workers=4,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset,
)

trainer.train()

model.save_pretrained("tinyLLM")
tokenizer.save_pretrained("tinyLLM")
