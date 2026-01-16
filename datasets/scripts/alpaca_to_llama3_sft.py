from datasets import load_dataset

N = 200

ds = load_dataset("tatsu-lab/alpaca", split="train")
ds = ds.shuffle(seed=42).select(range(N))

def format_sample(ex):
    sys = "You are a helpful assistant. Answer concisely and correctly."
    user = ex["instruction"].strip()
    if ex.get("input") and ex["input"].strip():
        user += "\n" + ex["input"].strip()
    assistant = ex["output"].strip()

    text = "<SFT>"
    text += "<|start_header_id|>system<|end_header_id|>\n" + sys + "<|eot_id|>\n"
    text += "<|start_header_id|>user<|end_header_id|>\n" + user + "<|eot_id|>\n"
    text += "<|start_header_id|>assistant<|end_header_id|>\n" + assistant + "<|eot_id|>\n"
    return text

out_path = "datasets/processed/train.txt"
with open(out_path, "w", encoding="utf-8") as f:
    for ex in ds:
        f.write(format_sample(ex))
print(f"Wrote {out_path} with {N} samples")
