from transformers import pipeline

pipe = pipeline(
    "text-generation",
    model="./tinyLLM",
    tokenizer="./tinyLLM",
)

prompt = "User: Yo\nAssistant:"
out = pipe(prompt, max_new_tokens=10, do_sample=False)

print(out[0]["generated_text"])