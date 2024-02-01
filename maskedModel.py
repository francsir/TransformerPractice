from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch

model_checkpoint = "distilbert-base-uncased"
model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)


text = "This is a great [MASK]"

## Using the regular model
inputs = tokenizer(text, return_tensors="pt")
token_logits = model(**inputs).logits

mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]
mask_token_logits = token_logits[0, mask_token_index, :]

top_5_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()

for token in top_5_tokens:
	print(f"'>>>{text.replace(tokenizer.mask_token, tokenizer.decode([token]))}'")

## Using updated dataset to fine-tine to specific domain

from datasets import load_dataset

imdb_dataset = load_dataset("imdb")

def tokenize_function(examples):
	result = tokenizer(examples["text"])
	if tokenizer.is_fast:
		result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
	return result

tokenized_datasets = imdb_dataset.map(
	tokenize_function, batched = True, remove_columns=["text", "label"]
)

chunk_size = 128



def group_texts(examples):


	concatenated_examples = {k:sum(examples[k], []) for k in examples.keys()}
	
	total_length = len(concatenated_examples[list(examples.keys())[0]])

	total_length = (total_length//chunk_size) * chunk_size

	result = {
		k:[t[i:i+chunk_size] for i in range(0, total_length, chunk_size)]
		for k, t in concatenated_examples.items()
	}

	result["labels"] = result["input_ids"].copy()
	return result

lm_dataset = tokenized_datasets.map(group_texts, batched=True)
print(lm_dataset)
