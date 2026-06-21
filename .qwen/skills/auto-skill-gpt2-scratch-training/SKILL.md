---
name: gpt2-scratch-training
description: Train a GPT-2-style causal LM from scratch on a plain-text corpus — chunk tokenized text into fixed-length blocks, scale model config to dataset size, and avoid the silent-truncation trap.
source: auto-skill
extracted_at: '2026-06-21T10:46:06.350Z'
---

# Train a GPT-2-style causal LM from scratch

Use when the user wants to train a small GPT-2 architecture (via `transformers` `GPT2LMHeadModel`) from scratch on a raw text corpus (`train.txt` or similar), rather than fine-tuning a pretrained model.

## The silent-truncation trap (critical)

**Never** load a full text corpus with `tokenizer(text, truncation=True)`. GPT-2's default `max_length` is 1024 tokens, so this silently truncates the entire dataset to the first 1024 tokens and discards everything else. On an 8 MB / 2.2M-token file this means training on ~0.05% of the data with no warning.

**Correct approach**: tokenize the full text *without* truncation, then chunk into fixed-length blocks:

```python
BLOCK_SIZE = 1024  # match n_positions / n_ctx in the model config

all_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
n_blocks = len(all_ids) // BLOCK_SIZE
trimmed_ids = all_ids[: n_blocks * BLOCK_SIZE]
input_ids = [trimmed_ids[i : i + BLOCK_SIZE] for i in range(0, len(trimmed_ids), BLOCK_SIZE)]

dataset = Dataset.from_dict({
    "input_ids": input_ids,
    "attention_mask": [[1] * BLOCK_SIZE for _ in input_ids],
    "labels": [ids[:] for ids in input_ids],
})
```

For a plain causal-LM corpus, `labels` is just a copy of `input_ids` (no prompt masking, no -100). Prompt→completion masking is only for instruction-tuning formats where the data is actually structured as prompt/completion pairs — do not invent that structure for raw prose.

## Scaling model config to dataset size

Match parameter count to token count so the model has enough capacity without overfitting a small dataset or underfitting a large one:

| Dataset size (tokens) | Approx params | n_embd | n_layer | n_head | n_positions |
|---|---|---|---|---|---|
| ~1–5M   | ~25–85M  | 384–768 | 6–12  | 6–12  | 256–512 |
| ~20–100M| ~350M    | 1024    | 24    | 16    | 1024 |
| ~100M+  | ~350M–1B| 1024–1600 | 24–48 | 16–25 | 1024 |

Token estimate: `chars / 4` is a rough proxy; exact count requires running the tokenizer. For a file of N MB, expect roughly N × 250k tokens for English prose.

## Training args for from-scratch training

- `learning_rate`: 3e-4 for ~100M+ params (lower than the 5e-4 typical for very small models)
- `warmup_steps`: ~200 prevents early-training loss spikes on larger models
- `num_train_epochs`: 1–3 for ≥20M-token datasets; more epochs risk overfitting. 200 epochs only makes sense for tiny (<100k-token) datasets.
- `fp16`: True if a CUDA GPU is available; set False on CPU.
- `per_device_train_batch_size`: 4 for ~350M @ 1024 ctx on a typical GPU; lower if OOM.

## Dataset sourcing

For a cleaned English Wikipedia corpus, use HuggingFace `datasets`:

```python
from datasets import load_dataset
ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")  # ~100M tokens
with open("train.txt", "w", encoding="utf-8") as f:
    for ex in ds:
        if ex["text"].strip():
            f.write(ex["text"])
            if not ex["text"].endswith("\n"):
                f.write("\n")
```

The raw variant (`wikitext-103-raw-v1`) preserves original casing and punctuation; the `wikitext-103-v1` variant applies lowercasing.

## Verification checklist

Before launching a long run, confirm:
1. Print `len(dataset)` and `BLOCK_SIZE` — token count should be in the right ballpark vs. file size, not 1024.
2. Print `print_parameter_count(model)` — should match the intended scale.
3. The `wandb.init` run name reflects the actual param count (e.g. `tinyLLM-350M-run`), not a stale number.