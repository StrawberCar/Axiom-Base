from datasets import load_dataset

# WikiText-103-raw-v1: cleaned plain-text extraction of Wikipedia articles
# (~520 MB, ~100M tokens). Raw version keeps original casing and punctuation.
dataset = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")

with open("train.txt", "w", encoding="utf-8") as f:
    for example in dataset:
        text = example["text"]
        if text.strip():
            f.write(text)
            if not text.endswith("\n"):
                f.write("\n")

print(f"Saved {len(dataset):,} lines to train.txt")