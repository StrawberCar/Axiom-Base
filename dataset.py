from datasets import load_dataset

# WikiText-103-raw-v1: cleaned plain-text extraction of Wikipedia articles
# (~520 MB, ~100M tokens). Raw version keeps original casing and punctuation.
print("Downloading dataset...", flush=True)
dataset = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
print(f"Dataset loaded: {len(dataset):,} examples", flush=True)

written = 0
with open("train.txt", "w", encoding="utf-8") as f:
    for i, example in enumerate(dataset):
        text = example["text"]
        if text.strip():
            f.write(text)
            if not text.endswith("\n"):
                f.write("\n")
            written += 1
        if (i + 1) % 100000 == 0:
            print(f"  Processed {i + 1:,}/{len(dataset):,} examples ({written:,} written)", flush=True)

print(f"Saved {written:,} non-empty lines to train.txt", flush=True)