from datasets import load_dataset

raw_dataset = load_dataset("glue", "mrpc")
print(raw_dataset)